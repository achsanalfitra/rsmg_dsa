use core::sync::atomic::{AtomicUsize, Ordering};
use std::hint::spin_loop;

use crate::handle::{AtomicHandle, AtomicHandleTrait};
extern crate alloc;
use alloc::alloc::{Layout, alloc, dealloc, handle_alloc_error};

static DEFAULT_LENGTH: usize = 8 * size_of::<*mut u8>();
static DEFAULT_CAPACITY: usize = 8 * size_of::<*mut u8>();
static DEFAULT_MULTIPLIER: usize = 2;
static WRITING_STATE: usize = 1;
static FREE_STATE: usize = 2;

struct ContiguousArray<T> {
    inner: AtomicHandle<*mut InnerContiguousArray<T>>,
}

struct InnerContiguousArray<T> {
    address: *mut *mut AtomicHandle<*mut T>,
    meta: ContiguousArrayMetadata,
}

struct ContiguousArrayMetadata {
    length: usize,
    capacity: usize,
    locker_count: AtomicUsize,
    resizing: AtomicUsize,
}
/// Rules of this impl:
///
/// Major ops include [`push`, `pop`] and many to come.
/// By definition, major ops may resize and reallocate the address.
/// This makes reading internal pointers dangerous.
///
/// Local ops are those that read internal pointers of this structure
/// including [`get`, `inspect_element`].
///
/// The rule is that local ops only care about length.
/// If length check succeeds, they increment locker_count by 1,
/// perform their read, then decrement locker_count.
///
/// Major ops spin on locker_count. They wait until locker_count
/// reaches zero before locking the whole structure through
/// `update_exclusive`, then once done, release it.
impl<T> ContiguousArray<T> {
    fn new() -> Self {
        let inner_array = InnerContiguousArray::<T>::new();

        let inner_ptr = Box::into_raw(Box::new(inner_array));

        Self {
            inner: AtomicHandle::new(inner_ptr),
        }
    }

    fn push(&self, data: T) {
        let data_ptr = Box::into_raw(Box::new(data));
        let data_handle = Box::into_raw(Box::new(AtomicHandle::new(data_ptr)));

        self.inner
            .update_exclusive(|inner_ptr: *mut InnerContiguousArray<T>| unsafe {
                // This wins, when it enters
                let inner = inner_ptr.as_mut().unwrap();

                // this is currently not behaving like what I imagined
                // the whole thing shouldn't be an update_exclusive()
                // only reallocate is update_exclusive()
                // the correct approach is to check periodically (spinlock)
                // on the locker_count then do massive operation
                // when locker_count reaches 0
                // we don't need any other metadata since
                // massive operations are, by default, update_exclusive

                // these only need to check for locker count before doing the operation

                loop {
                    if core::hint::black_box(inner.meta.locker_count.load(Ordering::Relaxed) > 0) {
                        core::hint::spin_loop();
                        continue;
                    }

                    let current_length = inner.meta.length;
                    let current_capacity = inner.meta.capacity;

                    inner.meta.resizing.store(WRITING_STATE, Ordering::SeqCst);
                    if current_length + 1 > current_capacity {
                        inner.reallocate();
                    }

                    inner.push(data_handle);
                    inner.meta.resizing.store(FREE_STATE, Ordering::Release);
                    break;
                }

                inner_ptr
            });
    }

    fn pop(&self) -> Result<Option<T>, Box<dyn std::error::Error>> {
        let result = core::cell::Cell::new(Ok(None));

        self.inner
            .update_exclusive(|inner_ptr: *mut InnerContiguousArray<T>| unsafe {
                let inner = inner_ptr.as_mut().unwrap();

                loop {
                    // 1. Check for active inspectors
                    if core::hint::black_box(inner.meta.locker_count.load(Ordering::Relaxed) > 0) {
                        core::hint::spin_loop();
                        continue;
                    }

                    // 2. Lock the state for massive operation
                    inner.meta.resizing.store(WRITING_STATE, Ordering::SeqCst);

                    // 3. Perform the actual pop
                    // We do this inside the locker drain to ensure no one is
                    // reading the tail element while we drop it.
                    let popped_data = inner.pop();
                    result.set(popped_data);

                    // 4. Release the state
                    inner.meta.resizing.store(FREE_STATE, Ordering::Release);
                    break;
                }

                inner_ptr
            });

        result.into_inner()
    }

    // fn pop(&self) -> Result<Option<T>, Box<dyn std::error::Error>> {
    //     let result = core::cell::Cell::new(Ok(None));

    //     self.inner.update_exclusive(|inner| unsafe {
    //         let popped_data = inner.as_mut().unwrap().pop();
    //         result.set(popped_data);
    //         inner
    //     });

    //     result.into_inner()
    // }

    fn len(&self) -> usize {
        let result = core::cell::Cell::new(0usize);

        self.inner.update_exclusive(|inner| unsafe {
            result.set(core::ptr::read(inner).len());
            inner
        });

        result.into_inner()
    }

    fn get(&self, index: usize) -> Option<&T> {
        let result = core::cell::Cell::new(None);
        let element = core::cell::Cell::new(None);
        let len = core::cell::Cell::new(0usize);

        self.inner.update(|inner| unsafe {
            len.set(inner.as_ref().unwrap().len());
            inner
        });

        if len.into_inner() < index {
            return None;
        }

        // if the length check is done, I know that I am now a reader
        self.inner.update(|inner| unsafe {
            inner
                .as_mut()
                .unwrap()
                .meta
                .locker_count
                .fetch_add(1, Ordering::Acquire);
            element.set(inner.as_ref().unwrap().get(index));
            inner
        });

        // look this now provide the read only data instead for convenience
        // but the vision is there, the inner atomic handle is propped up
        // then modified to have strong lock

        unsafe {
            element
                .into_inner()
                .unwrap()
                .as_mut()
                .unwrap()
                .update_exclusive(|element| {
                    result.set(element.as_ref());
                    element
                });
        }

        // give back the ordering
        self.inner.update(|inner| unsafe {
            inner
                .as_mut()
                .unwrap()
                .meta
                .locker_count
                .fetch_sub(1, Ordering::Release);
            inner
        });

        result.into_inner()
    }

    fn inspect_element(&self, index: usize, f: impl Fn(&mut T)) {
        self.inner
            .update_exclusive(|inner_ptr: *mut InnerContiguousArray<T>| unsafe {
                let inner = inner_ptr.as_mut().unwrap();
                loop {
                    // 1. Wait for massive operations (push/pop/reallocate) to finish intent
                    if core::hint::black_box(
                        inner.meta.resizing.load(Ordering::Acquire) == WRITING_STATE,
                    ) {
                        spin_loop();
                        continue;
                    }

                    // 2. Register yourself as a reader
                    inner.meta.locker_count.fetch_add(1, Ordering::SeqCst);

                    // 3. THE CRITICAL RE-CHECK: Now that we are pinned, is the index still valid?
                    // A 'pop' could have finished right before we did the fetch_add.
                    if index >= inner.meta.length {
                        inner.meta.locker_count.fetch_sub(1, Ordering::SeqCst);
                        return inner_ptr; // Early exit: index is now out of bounds
                    }

                    // 4. Access the element
                    // We use .get(index) safely here because we verified index < length
                    // while holding the locker_count pin.
                    if let Some(handle_ptr) = inner.get(index) {
                        handle_ptr.as_mut().unwrap().update_exclusive(|element| {
                            f(&mut *element);
                            element
                        });
                    }

                    // 5. Unpin and finish
                    inner.meta.locker_count.fetch_sub(1, Ordering::SeqCst);
                    break;
                }

                inner_ptr
            });
    }

    // fn inspect_element(&self, index: usize, f: impl Fn(&mut T)) {
    //     let len = core::cell::Cell::new(0usize);

    //     self.inner.update(|inner| unsafe {
    //         len.set(inner.as_ref().unwrap().len());
    //         inner
    //     });

    //     if len.into_inner() < index {
    //         return;
    //     }

    //     self.inner
    //         .update_exclusive(|inner_ptr: *mut InnerContiguousArray<T>| unsafe {
    //             let inner = inner_ptr.as_mut().unwrap();
    //             loop {
    //                 if core::hint::black_box(
    //                     inner.meta.resizing.load(Ordering::Acquire) == WRITING_STATE,
    //                 ) {
    //                     spin_loop();
    //                     continue;
    //                 }

    //                 inner.meta.locker_count.fetch_add(1, Ordering::SeqCst);
    //                 inner
    //                     .get(index)
    //                     .unwrap()
    //                     .as_mut()
    //                     .unwrap()
    //                     .update_exclusive(|element| {
    //                         f(&mut *element);
    //                         element
    //                     });
    //                 inner.meta.locker_count.fetch_sub(1, Ordering::SeqCst);
    //                 break;
    //             }

    //             inner
    //         });
    // }
}

impl<T> InnerContiguousArray<T> {
    fn new() -> Self {
        let layout = Layout::array::<*mut AtomicHandle<*mut T>>(DEFAULT_CAPACITY).unwrap();

        let allocated_buffer = unsafe {
            let ptr = alloc(layout) as *mut *mut AtomicHandle<*mut T>;

            if ptr.is_null() {
                handle_alloc_error(layout);
            }

            ptr
        };

        Self {
            address: allocated_buffer,
            meta: ContiguousArrayMetadata::new(),
        }
    }

    fn push(&mut self, data: *mut AtomicHandle<*mut T>) {
        unsafe {
            let base = self.address;
            base.add(self.meta.length).write(data);
            self.meta.length += 1;

            let idx = self.meta.length - 1;
            let handle_ptr = base.add(idx).read();

            assert!(!handle_ptr.is_null());

            let pushed_t = &*handle_ptr;
            let (_, n) = pushed_t.get();
            assert!(n != 0);
        }
    }

    fn pop(&mut self) -> Result<Option<T>, Box<dyn std::error::Error>> {
        if self.meta.length < 1 {
            return Ok(None);
        }

        unsafe {
            let base = self.address;
            let top = base.add(self.meta.length - 1).read();

            if top.is_null() {
                return Ok(None);
            }

            self.meta.length -= 1;
            let top_handle = Box::from_raw(top);
            let (data_ptr, _) = top_handle.get();
            let data = core::ptr::read(data_ptr);

            drop(Box::from_raw(data_ptr));

            return Ok(Some(data));
        }
    }

    fn reallocate(&mut self) {
        let layout =
            Layout::array::<*mut AtomicHandle<*mut T>>(self.meta.capacity * DEFAULT_MULTIPLIER)
                .unwrap();

        let allocated_buffer = unsafe {
            let ptr = alloc(layout) as *mut *mut AtomicHandle<*mut T>;

            if ptr.is_null() {
                handle_alloc_error(layout);
            }

            ptr
        };

        unsafe {
            core::ptr::copy_nonoverlapping(self.address, allocated_buffer, self.meta.length);

            let old_layout =
                Layout::array::<*mut AtomicHandle<*mut T>>(self.meta.capacity).unwrap();
            dealloc(self.address as *mut u8, old_layout);
        }

        self.address = allocated_buffer;

        self.meta.capacity *= DEFAULT_MULTIPLIER;
    }

    fn len(&self) -> usize {
        self.meta.length
    }

    fn get(&self, index: usize) -> Option<*mut AtomicHandle<*mut T>> {
        // guarantee to exist

        let base = self.address;

        unsafe {
            let target = base.add(index).read();
            let target_handle = target.as_mut().unwrap();
            Some(target_handle)
        }
    }
}

unsafe impl<T: Copy + Send> Sync for ContiguousArray<T> {}
unsafe impl<T: Copy + Send> Send for ContiguousArray<T> {}

impl ContiguousArrayMetadata {
    fn new() -> Self {
        Self {
            length: 0,
            capacity: DEFAULT_CAPACITY,
            locker_count: AtomicUsize::new(0),
            resizing: AtomicUsize::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::prim::array::ContiguousArray;

    #[test]
    fn test_simple_push() {
        let array: ContiguousArray<usize> = ContiguousArray::new();
        let data = 1;
        array.push(data);
    }

    #[test]
    fn test_reallocate_push() {
        let array: ContiguousArray<usize> = ContiguousArray::new();
        let data = 1;

        for _ in 0..10 {
            array.push(data);
        }
    }

    #[test]
    fn test_push_pop_single() {
        let array: ContiguousArray<usize> = ContiguousArray::new();
        let data = 1;
        array.push(data);

        assert_eq!(array.pop().unwrap().unwrap(), data);
    }

    #[test]
    fn test_push_pop_integrity() {
        let array: ContiguousArray<usize> = ContiguousArray::new();

        for i in 0..10 {
            array.push(i);
        }

        for i in 0..10 {
            assert_eq!(array.pop().unwrap().unwrap(), (9 - i));
        }
    }

    #[test]
    fn test_integrity_concurrent_push() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let mut handles = vec![];
        let num_threads = 10;
        let ops_per_thread = 1000;

        for t in 0..num_threads {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..ops_per_thread {
                    a.push(t * 10000 + i);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let mut results = vec![];
        while let Ok(Some(val)) = array.pop() {
            results.push(val);
        }

        // Verify total volume
        assert_eq!(results.len(), num_threads * ops_per_thread);

        // Verify per-thread LIFO integrity
        for t in 0..num_threads {
            let thread_prefix = t * 10000;
            let mut thread_values: Vec<usize> = results
                .iter()
                .filter(|&&v| v >= thread_prefix && v < thread_prefix + 1000)
                .cloned()
                .collect();

            let mut expected: Vec<usize> = (0..ops_per_thread)
                .map(|i| thread_prefix + i)
                .rev()
                .collect();

            // Values within a specific thread's context must be popped in reverse order
            assert_eq!(thread_values, expected);
        }
    }

    #[test]
    fn test_concurrency_zero_sum_hammer() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let mut handles = vec![];
        let num_threads = 10;
        let ops_per_thread = 1000;

        for t in 0..num_threads {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..ops_per_thread {
                    if i % 2 == 0 {
                        a.push(t * 1000 + i);
                    } else {
                        let _ = a.pop();
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Must be exactly zero because pushes == pops per thread
        let mut final_count = 0;
        while let Ok(Some(_)) = array.pop() {
            final_count += 1;
        }

        assert_eq!(
            final_count, 0,
            "Stack leaked or corrupted during zero-sum hammer"
        );
    }

    #[test]
    fn test_concurrency_stable_read_hammer() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let num_readers = 12;
        let entries = 10_000;

        // --- PHASE 1: PUSH MANY FIRST ---
        // Single-threaded or sequential push to establish ground truth data
        // without triggering reallocation during the read phase.
        for i in 0..entries {
            array.push(i);
        }

        // Ensure the array actually has the data
        assert_eq!(
            array.len(),
            entries,
            "Array failed to initialize for hammer"
        );

        let mut handles = vec![];

        // --- PHASE 2: CONCURRENCY READ HAMMER ---
        // All threads hammering 'get' on a stable memory block.
        // This tests if 'update_exclusive' handles the high-frequency
        // "read-only reference" vision correctly.
        for t in 0..num_readers {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..entries {
                    // Stress different indices to ensure the pointer math
                    // inside your 'inner' handle is consistent across threads.
                    let target_idx = (i + t) % entries;

                    if let Some(val_ref) = a.get(target_idx) {
                        let val = *val_ref;

                        // Verify the data is actually what we pushed
                        // If this fails, the 'strong lock' is leaking state.
                        assert_eq!(val, target_idx, "Data corruption at index {}", target_idx);

                        std::hint::black_box(val);
                    } else {
                        panic!("Ground Truth Failure: Index {} should exist", target_idx);
                    }
                }
            }));
        }

        for h in handles {
            h.join()
                .expect("Reader thread panicked - possible memory safety violation");
        }

        println!(
            "Stable Read Hammer passed for {} concurrent readers.",
            num_readers
        );
    }

    #[test]
    fn test_concurrency_element_rw_race_hammer() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let entries = 1000;
        let ops_per_thread = 5000;

        // Phase 1: Initialize with zeroed data
        for _ in 0..entries {
            array.push(0);
        }

        let mut handles = vec![];

        // --- WRITER THREADS (The Mutators) ---
        for _ in 0..4 {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let idx = i % entries;
                    a.inspect_element(idx, |val| {
                        *val += 1;
                    });
                }
            }));
        }

        // --- READER THREADS (The Observers) ---
        // Now using inspect_element for safe, non-torn reads.
        for _ in 0..4 {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let idx = i % entries;
                    // Reading through the same closure API to ensure atomicity
                    a.inspect_element(idx, |val| {
                        let data = *val;
                        std::hint::black_box(data);
                    });
                }
            }));
        }

        for h in handles {
            h.join()
                .expect("Race Hammer failed: Element-scoped lock violation or Deadlock");
        }

        // --- FINAL GROUND TRUTH CHECK ---
        let total_sum = std::sync::atomic::AtomicUsize::new(0);

        for i in 0..entries {
            // We use a reference to the atomic so the closure can 'capture' it
            // and perform the addition safely.
            array.inspect_element(i, |val| {
                total_sum.fetch_add(*val, std::sync::atomic::Ordering::SeqCst);
            });
        }

        let final_val = total_sum.load(std::sync::atomic::Ordering::SeqCst);

        assert_eq!(
            final_val,
            4 * ops_per_thread,
            "Lost updates! Element-scoped locking is leaky. Expected {}, got {}",
            4 * ops_per_thread,
            final_val
        );

        println!("Element RW Race Hammer passed. Total sum: {}", final_val);
    }

    #[test]
    fn test_ghost_isolation_no_resizing() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let entries = 1000;
        let ops_per_thread = 5000;

        // Pre-allocate enough capacity so reallocate() is NEVER called during the test
        // If this passes, the bug is 100% inside your reallocate/pop logic.
        for _ in 0..entries {
            array.push(0);
        }

        let mut handles = vec![];
        for _ in 0..8 {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..ops_per_thread {
                    a.inspect_element(i % entries, |val| {
                        *val += 1;
                    });
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let total = get_total_sum_atomic(&array, entries);
        assert_eq!(
            total,
            8 * ops_per_thread,
            "Ghost found even WITHOUT resizing! The issue is in inspect_element itself."
        );
    }

    fn get_total_sum_atomic(array: &ContiguousArray<usize>, entries: usize) -> usize {
        let sum = std::sync::atomic::AtomicUsize::new(0);
        for i in 0..entries {
            array.inspect_element(i, |val| {
                sum.fetch_add(*val, std::sync::atomic::Ordering::SeqCst);
            });
        }
        sum.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[test]
    fn test_pointer_aliasing_signature() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let entries = 100;

        // Give every index a massive unique signature
        for i in 0..entries {
            array.push(i * 1_000_000);
        }

        // Hammer increments
        let a = Arc::clone(&array);
        let handle = std::thread::spawn(move || {
            for _ in 0..1000 {
                a.inspect_element(5, |val| *val += 1);
            }
        });
        handle.join().unwrap();

        // Verify signatures
        for i in 0..entries {
            array.inspect_element(i, |val| {
                let expected_base = i * 1_000_000;
                if i == 5 {
                    assert_eq!(*val, expected_base + 1000);
                } else {
                    // If this fails, Index 5's increment "leaked" into Index i
                    assert_eq!(*val, expected_base, "Aliasing detected at Index {}", i);
                }
            });
        }
    }

    #[test]
    fn test_metadata_consistency_hammer() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        array.push(0);

        let a1 = Arc::clone(&array);
        let h1 = std::thread::spawn(move || {
            for _ in 0..100_000 {
                // Simulate an inspector
                a1.inspect_element(0, |v| {
                    std::hint::black_box(*v);
                });
            }
        });

        let a2 = Arc::clone(&array);
        let h2 = std::thread::spawn(move || {
            for _ in 0..1000 {
                // Simulate a massive op (push/reallocate)
                a2.push(1);
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();
        // If the internal assertions or spin-loops didn't deadlock/crash,
        // metadata consistency is likely okay.
    }

    #[test]
    fn test_contiguous_array_push_pop_integrity_hammer() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let initial_count = 100;
        let ops = 5000;

        // Phase 1: Seed the array with recognizable data
        for i in 0..initial_count {
            array.push(i);
        }

        let mut handles = vec![];

        // --- THE CHURN THREAD (Push & Pop) ---
        // This thread triggers reallocations and memory frees constantly.
        let churn_array = Arc::clone(&array);
        handles.push(std::thread::spawn(move || {
            for i in 0..ops {
                // Alternating push and pop to keep the array "shaking"
                churn_array.push(i + 1000);
                let _ = churn_array.pop();
            }
        }));

        // --- THE VERIFIER THREADS ---
        // These threads check that the "base" data (indices 0..50) never changes.
        // If reallocate is leaky, these will read garbage or crash.
        for _ in 0..3 {
            let verify_array = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for _ in 0..ops {
                    // We pick a stable index that isn't being popped
                    let target_idx = 10;
                    verify_array.inspect_element(target_idx, |val| {
                        // Index 10 was initialized to 10. It should STAY 10.
                        assert_eq!(*val, 10, "Memory corruption detected during push/pop!");
                    });
                }
            }));
        }

        for h in handles {
            h.join()
                .expect("Hammer failed! Possible Use-After-Free or Data Race.");
        }

        println!("Integrity Hammer passed: Data remained stable during heavy churn.");
    }

    #[test]
    fn test_rapid_growth() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let mut handles = vec![];
        let threads = 10;
        let ops_per_thread = 1000;

        for t in 0..threads {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..ops_per_thread {
                    // Testing high-frequency pushes to trigger DEFAULT_MULTIPLIER jumps
                    a.push(t * 100000 + i);
                }
            }));
        }

        for h in handles {
            h.join().expect("Thread panicked during rapid growth");
        }

        // Verify total count matches total pushes
        assert_eq!(array.len(), threads * ops_per_thread);

        // Verify no corruption by draining the array
        let mut pop_count = 0;
        while let Some(_) = array.pop().expect("Pop error") {
            pop_count += 1;
        }

        assert_eq!(pop_count, threads * ops_per_thread);
        println!(
            "Rapid growth passed: {} elements pushed and verified.",
            pop_count
        );
    }

    #[test]
    fn test_concurrent_inspect_same_index() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        array.push(0); // Initialize slot 0

        let mut handles = vec![];
        let threads = 10;
        let ops_per_thread = 1000;

        for _ in 0..threads {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for _ in 0..ops_per_thread {
                    // Both array lock and element lock are stressed here
                    a.inspect_element(0, |val| {
                        *val += 1;
                    });
                }
            }));
        }

        for h in handles {
            h.join().expect("Thread panicked during concurrent inspect");
        }

        let expected = threads * ops_per_thread;
        array.inspect_element(0, |val| {
            assert_eq!(
                *val, expected,
                "Lost updates! Expected {} but got {}. Element lock is leaky.",
                expected, *val
            );
        });
    }

    #[test]
    fn test_inspect_during_pop_of_same_index() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let target_idx = 9;
        let ops = 50;

        for i in 0..=target_idx {
            array.push(i);
        }

        let mut handles = vec![];

        // --- THE CHURNER ---
        let pop_array = Arc::clone(&array);
        handles.push(std::thread::spawn(move || {
            for _ in 0..ops {
                // Constantly shrinking and growing the boundary
                let _ = pop_array.pop();
                pop_array.push(99999);
            }
        }));

        // --- THE BOUNDARY INSPECTORS ---
        for _ in 0..3 {
            let inspect_array = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for _ in 0..ops {
                    let current_len = inspect_array.len();
                    if current_len == 0 {
                        continue;
                    }

                    let try_idx = current_len - 1;

                    // inspect_element should either run safely or return early via the bounds check
                    inspect_array.inspect_element(try_idx, |val| {
                        let v = *val;
                        let is_valid = v <= target_idx || v == 99999;
                        assert!(is_valid, "Corrupt read at index {}: {}", try_idx, v);
                    });

                    // Validate explicit bounds protection
                    // (Assuming inspect_element returns early for indices >= len)
                    inspect_array.inspect_element(1000000, |_| {
                        panic!("Boundary check failed: Inspected index far beyond len");
                    });
                }
            }));
        }

        for h in handles {
            h.join()
                .expect("Integrity failure: Race between inspect and pop/push");
        }
    }
}
