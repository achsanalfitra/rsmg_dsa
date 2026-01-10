use core::hint::spin_loop;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::handle::{AtomicHandle, AtomicHandleTrait};
extern crate alloc;
use alloc::alloc::{Layout, alloc, dealloc, handle_alloc_error};

static DEFAULT_CAPACITY: usize = 8 * size_of::<*mut u8>();
static DEFAULT_MULTIPLIER: usize = 2;
static WRITING_STATE: usize = 1;
static FREE_STATE: usize = 2;

pub struct ContiguousArray<T> {
    inner: AtomicHandle<*mut InnerContiguousArray<T>>,
}

struct InnerContiguousArray<T> {
    address: *mut *mut AtomicHandle<*mut T>,
    meta: ContiguousArrayMetadata,
}

struct ContiguousArrayMetadata {
    length: AtomicUsize,
    capacity: AtomicUsize,
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
/// including [`inspect_element`].
///
/// The rule is that local ops only care about length.
/// If length check succeeds, they increment locker_count by 1,
/// perform their read, then decrement locker_count.
///
/// Major ops spin on locker_count. They wait until locker_count
/// reaches zero before locking the whole structure through
/// `update_exclusive`, then once done, release it.
///
///
/// Implementation of ContiguousArray
///
/// # Example
///
/// ```
/// # use rsmg_core::prim::array::ContiguousArray;
/// # fn main() {
/// let array = ContiguousArray::<usize>::new();
/// let data = 1;
/// array.push(data);
///
/// // smuggle data out
/// // do not perform expensive operation inside
/// // inspect_element.
/// // Clone data outside if possible, or mutate
/// // in-place
/// let result = std::cell::Cell::new(0usize);
///
/// array.inspect_element(0, |element|
///     result.set(*element) // usize implements Copy, no need to clone()
/// );
///
/// assert_eq!(result.into_inner(), data);
///
/// array.pop();
///
/// assert_eq!(array.len(), 0);
/// # }
///
/// ```
impl<T> ContiguousArray<T> {
    pub fn new() -> Self {
        let inner_array = InnerContiguousArray::<T>::new();

        let inner_ptr = Box::into_raw(Box::new(inner_array));

        Self {
            inner: AtomicHandle::new(inner_ptr),
        }
    }

    pub fn push(&self, data: T) {
        let data_ptr = Box::into_raw(Box::new(data));
        let data_handle = Box::into_raw(Box::new(AtomicHandle::new(data_ptr)));

        unsafe {
            loop {
                let inner_ptr = self.inner.get().0;
                if inner_ptr.is_null() {
                    unreachable!("try to panic this");
                }

                if (*inner_ptr).meta.locker_count.load(Ordering::Acquire) > 0 {
                    core::hint::spin_loop();
                    continue;
                }

                if (*inner_ptr)
                    .meta
                    .resizing
                    .compare_exchange_weak(
                        FREE_STATE,
                        WRITING_STATE,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_err()
                {
                    core::hint::spin_loop();
                    continue;
                }

                if (*inner_ptr).meta.locker_count.load(Ordering::Acquire) > 0 {
                    (*inner_ptr)
                        .meta
                        .resizing
                        .store(FREE_STATE, Ordering::Release);
                    core::hint::spin_loop();
                    continue;
                }

                let length = (*inner_ptr).meta.length.load(Ordering::Acquire);
                let capacity = (*inner_ptr).meta.capacity.load(Ordering::Acquire);

                if length + 1 > capacity {
                    Self::reallocate_raw(inner_ptr);
                }

                let base = (*inner_ptr).address;
                base.add(length).write(data_handle);

                (*inner_ptr).meta.length.fetch_add(1, Ordering::Release);

                (*inner_ptr)
                    .meta
                    .resizing
                    .store(FREE_STATE, Ordering::Release);
                break;
            }
        }
    }

    pub fn pop(&self) -> Result<Option<T>, Box<dyn std::error::Error>> {
        unsafe {
            loop {
                let inner_ptr = self.inner.get().0;
                if inner_ptr.is_null() {
                    return Ok(None);
                }

                if (*inner_ptr).meta.locker_count.load(Ordering::Acquire) > 0 {
                    core::hint::spin_loop();
                    continue;
                }
                if (*inner_ptr)
                    .meta
                    .resizing
                    .compare_exchange_weak(
                        FREE_STATE,
                        WRITING_STATE,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_err()
                {
                    core::hint::spin_loop();
                    continue;
                }
                if (*inner_ptr).meta.locker_count.load(Ordering::Acquire) > 0 {
                    (*inner_ptr)
                        .meta
                        .resizing
                        .store(FREE_STATE, Ordering::Release);
                    core::hint::spin_loop();
                    continue;
                }

                let length = (*inner_ptr).meta.length.load(Ordering::Acquire);
                if length < 1 {
                    (*inner_ptr)
                        .meta
                        .resizing
                        .store(FREE_STATE, Ordering::Release);
                    return Ok(None);
                }

                let base = (*inner_ptr).address;
                let target_idx = length - 1;
                let handle_ptr = base.add(target_idx).read();

                if handle_ptr.is_null() {
                    (*inner_ptr)
                        .meta
                        .resizing
                        .store(FREE_STATE, Ordering::Release);
                    return Ok(None);
                }

                // do not sub yet if there are lingering readers
                // (*inner_ptr).meta.length.fetch_sub(1, Ordering::Release);

                while (*inner_ptr).meta.locker_count.load(Ordering::Acquire) > 0 {
                    core::hint::spin_loop();
                }

                (*inner_ptr).meta.length.fetch_sub(1, Ordering::Release);

                (*inner_ptr)
                    .meta
                    .resizing
                    .store(FREE_STATE, Ordering::Release);

                let handle_box = Box::from_raw(handle_ptr);
                let (data_ptr, _) = handle_box.get();
                let data = *Box::from_raw(data_ptr);

                return Ok(Some(data));
            }
        }
    }

    fn reallocate_raw(inner_ptr: *mut InnerContiguousArray<T>) {
        unsafe {
            let old_capacity = (*inner_ptr).meta.capacity.load(Ordering::Acquire);
            let new_capacity = old_capacity * DEFAULT_MULTIPLIER;
            let length = (*inner_ptr).meta.length.load(Ordering::Acquire);

            let layout = Layout::array::<*mut AtomicHandle<*mut T>>(new_capacity).unwrap();

            let new_buffer = {
                let ptr = alloc(layout) as *mut *mut AtomicHandle<*mut T>;
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                ptr
            };

            core::ptr::copy_nonoverlapping((*inner_ptr).address, new_buffer, length);
            let old_buffer = (*inner_ptr).address;
            (*inner_ptr).address = new_buffer;

            (*inner_ptr)
                .meta
                .capacity
                .store(new_capacity, Ordering::Release);

            let old_layout = Layout::array::<*mut AtomicHandle<*mut T>>(old_capacity).unwrap();

            dealloc(old_buffer as *mut u8, old_layout);
        }
    }

    pub fn len(&self) -> usize {
        unsafe {
            let inner_ptr = self.inner.get().0;
            if inner_ptr.is_null() {
                return 0;
            }

            loop {
                let resizing = (*inner_ptr).meta.resizing.load(Ordering::Acquire);
                if resizing == WRITING_STATE {
                    spin_loop();
                    continue;
                }

                let length = (*inner_ptr).meta.length.load(Ordering::Acquire);

                let resizing_after = (*inner_ptr).meta.resizing.load(Ordering::Acquire);
                if resizing_after == WRITING_STATE {
                    spin_loop();
                    continue;
                }

                return length;
            }
        }
    }

    pub fn inspect_element(&self, index: usize, f: impl Fn(&mut T)) {
        unsafe {
            let inner_ptr = self.inner.get().0;
            if inner_ptr.is_null() {
                return;
            }

            loop {
                let resizing = (*inner_ptr).meta.resizing.load(Ordering::Acquire);
                if resizing == WRITING_STATE {
                    spin_loop();
                    continue;
                }

                (*inner_ptr)
                    .meta
                    .locker_count
                    .fetch_add(1, Ordering::SeqCst);

                if (*inner_ptr).meta.resizing.load(Ordering::SeqCst) == WRITING_STATE {
                    (*inner_ptr)
                        .meta
                        .locker_count
                        .fetch_sub(1, Ordering::SeqCst);
                    spin_loop();
                    continue;
                }

                break;
            }

            let length = (*inner_ptr).meta.length.load(Ordering::Acquire);

            if index < length {
                let base = (*inner_ptr).address;
                let handle_ptr: *mut AtomicHandle<*mut T> = *base.add(index);

                if !handle_ptr.is_null() {
                    (*handle_ptr).update_exclusive(|element_ptr| {
                        if !element_ptr.is_null() {
                            f(&mut *element_ptr);
                        }
                        element_ptr
                    });
                }
            }

            (*inner_ptr)
                .meta
                .locker_count
                .fetch_sub(1, Ordering::Release);
        }
    }
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

            let current_len = self.meta.length.load(Ordering::Acquire);

            base.add(current_len).write(data);

            self.meta.length.fetch_add(1, Ordering::Release);

            let handle_ptr = base.add(current_len).read();
            assert!(!handle_ptr.is_null());

            let pushed_t = &*handle_ptr;
            let (_, n) = pushed_t.get();
            assert!(n != 0);
        }
    }

    fn pop(&mut self) -> Result<Option<T>, Box<dyn std::error::Error>> {
        let current_len = self.meta.length.load(Ordering::Acquire);
        if current_len < 1 {
            return Ok(None);
        }

        unsafe {
            let old_len = self.meta.length.fetch_sub(1, Ordering::AcqRel);
            let target_idx = old_len - 1;

            let base = self.address;
            let top = base.add(target_idx).read();

            if top.is_null() {
                unreachable!("try to panic this")
            }

            let top_handle = Box::from_raw(top);
            let (data_ptr, _) = top_handle.get();

            let data = core::ptr::read(data_ptr);

            drop(Box::from_raw(data_ptr));

            Ok(Some(data))
        }
    }
}

impl<T> Drop for ContiguousArray<T> {
    fn drop(&mut self) {
        unsafe {
            let (inner_ptr, _) = self.inner.get();
            if inner_ptr.is_null() {
                return;
            }
            let inner = &*inner_ptr;
            std::sync::atomic::fence(Ordering::Acquire);

            while inner
                .meta
                .resizing
                .compare_exchange_weak(
                    FREE_STATE,
                    WRITING_STATE,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                core::hint::spin_loop();
            }

            while inner.meta.locker_count.load(Ordering::Acquire) > 0 {
                core::hint::spin_loop();
            }

            let _ = Box::from_raw(inner_ptr);
        }
    }
}

impl<T> Drop for InnerContiguousArray<T> {
    fn drop(&mut self) {
        if self.meta.locker_count.load(Ordering::Acquire) > 0 {
            return;
        }

        let current_len = self.meta.length.load(Ordering::Acquire);
        let current_cap = self.meta.capacity.load(Ordering::Acquire);

        unsafe {
            let base_ptr = self.address;

            for i in 0..current_len {
                let handle_ptr: *mut AtomicHandle<*mut T> = *base_ptr.add(i);

                if !handle_ptr.is_null() {
                    let (data_ptr, _) = (*handle_ptr).get();

                    if !data_ptr.is_null() {
                        let _ = Box::from_raw(data_ptr);
                    }

                    let _ = Box::from_raw(handle_ptr);
                }
            }

            if current_cap > 0 {
                let spine_layout = Layout::array::<*mut AtomicHandle<*mut T>>(current_cap).unwrap();

                dealloc(self.address as *mut u8, spine_layout);
            }
        }
    }
}
unsafe impl<T: Copy + Send> Sync for ContiguousArray<T> {}
unsafe impl<T: Copy + Send> Send for ContiguousArray<T> {}

impl ContiguousArrayMetadata {
    fn new() -> Self {
        Self {
            length: AtomicUsize::new(0),
            capacity: AtomicUsize::new(DEFAULT_CAPACITY),
            locker_count: AtomicUsize::new(0),
            resizing: AtomicUsize::new(FREE_STATE),
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
        let ops_per_thread = 10;

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
            let thread_values: Vec<usize> = results
                .iter()
                .filter(|&&v| v >= thread_prefix && v < thread_prefix + 1000)
                .cloned()
                .collect();

            let expected: Vec<usize> = (0..ops_per_thread)
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
        let ops_per_thread = 50;

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
        // We use Arc for the test harness, but Rsmg-core handles the internal safety
        let array = std::sync::Arc::new(ContiguousArray::<usize>::new());
        let num_readers = 12;
        let entries = 100;

        // --- PHASE 1: PUSH MANY FIRST ---
        for i in 0..entries {
            array.push(i);
        }

        assert_eq!(
            array.len(),
            entries,
            "Array failed to initialize for hammer"
        );

        let mut handles = vec![];

        // --- PHASE 2: CONCURRENCY READ HAMMER ---
        // Using inspect_element ensures that the locker_count is incremented
        // and decremented correctly across high-frequency concurrent calls.
        println!("entering phase2");
        for t in 0..num_readers {
            let a = std::sync::Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..entries {
                    let target_idx = (i + t) % entries;

                    // inspect_element uses the closure-based lock vision
                    a.inspect_element(target_idx, |val| {
                        // Verify the data inside the exclusive scope
                        assert_eq!(
                            *val, target_idx,
                            "Data corruption at index {}! Thread {} saw wrong value.",
                            target_idx, t
                        );

                        // Prevent compiler from optimizing away the read
                        // std::hint::black_box(*val);
                    });
                }
            }));
        }

        for h in handles {
            h.join().expect(
                "Reader thread panicked - possible memory safety violation or race condition",
            );
        }

        println!(
            "Stable Read Hammer passed for {} concurrent readers using inspect_element.",
            num_readers
        );

        // At the very bottom of the test function
        std::mem::drop(array);
    }

    #[test]
    fn test_concurrency_element_rw_race_hammer() {
        let array = Arc::new(ContiguousArray::<usize>::new());
        let entries = 10;
        let ops_per_thread = 50;

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
        let entries = 10;
        let ops_per_thread = 50;

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
            for _ in 0..100 {
                // Simulate an inspector
                a1.inspect_element(0, |v| {
                    std::hint::black_box(*v);
                });
            }
        });

        let a2 = Arc::clone(&array);
        let h2 = std::thread::spawn(move || {
            for _ in 0..100 {
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
        let ops = 10;

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
        let ops_per_thread = 50;

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
        let ops_per_thread = 10;

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
