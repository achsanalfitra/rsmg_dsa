use crate::handle::{AtomicHandle, AtomicHandleTrait};
extern crate alloc;
use alloc::alloc::{Layout, alloc, dealloc, handle_alloc_error};

static DEFAULT_LENGTH: usize = 8 * size_of::<*mut u8>();
static DEFAULT_CAPACITY: usize = 8 * size_of::<*mut u8>();
static DEFAULT_MULTIPLIER: usize = 2;

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
    locker_count: usize,
}

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
                let inner = inner_ptr.as_mut().unwrap();
                let current_length = inner.meta.length;
                let current_capacity = inner.meta.capacity;

                // this is currently not behaving like what I imagined
                // the whole thing shouldn't be an update_exclusive()
                // only reallocate is update_exclusive()
                // the correct approach is to check periodically (spinlock)
                // on the locker_count then do massive operation
                // when locker_count reaches 0
                // we don't need any other metadata since
                // massive operations are, by default, update_exclusive
                if current_length + 1 > current_capacity {
                    inner.reallocate();
                }

                inner.push(data_handle);
                inner_ptr
            });
    }

    fn pop(&self) -> Result<Option<T>, Box<dyn std::error::Error>> {
        let result = core::cell::Cell::new(Ok(None));

        self.inner.update_exclusive(|inner| unsafe {
            let popped_data = inner.as_mut().unwrap().pop();
            result.set(popped_data);
            inner
        });

        result.into_inner()
    }

    fn len(&self) -> usize {
        let result = core::cell::Cell::new(0usize);

        self.inner.update(|inner| unsafe {
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

        self.inner.update(|inner| unsafe {
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

        result.into_inner()
    }

    fn inspect_element(&self, index: usize, f: impl Fn(&mut T)) {
        let element = core::cell::Cell::new(None);
        let len = core::cell::Cell::new(0usize);

        self.inner.update(|inner| unsafe {
            len.set(inner.as_ref().unwrap().len());
            inner
        });

        if len.into_inner() < index {
            return;
        }

        self.inner.update(|inner| unsafe {
            element.set(inner.as_ref().unwrap().get(index));
            inner
        });

        unsafe {
            element
                .into_inner()
                .unwrap()
                .as_mut()
                .unwrap()
                .update_exclusive(|element| {
                    f(&mut *element);
                    element
                });
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
            locker_count: 0,
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
        // These threads use inspect_element to increment values.
        // This tests if the &mut T remains exclusive.
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
        // These threads use get() to read the values.
        // They verify that they never see "torn" or garbage data.
        for _ in 0..4 {
            let a = Arc::clone(&array);
            handles.push(std::thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let idx = i % entries;
                    if let Some(val_ref) = a.get(idx) {
                        let val = *val_ref;
                        // In Alpha 1, we just ensure the read is safe/non-crashing.
                        std::hint::black_box(val);
                    }
                }
            }));
        }

        for h in handles {
            h.join()
                .expect("Race Hammer failed: Element-scoped lock violation");
        }

        // --- FINAL GROUND TRUTH CHECK ---
        // Total sum should be (Writers * Ops)
        let mut total_sum = 0;
        for i in 0..entries {
            if let Some(val) = array.get(i) {
                total_sum += *val;
            }
        }

        assert_eq!(
            total_sum,
            4 * ops_per_thread,
            "Lost updates! Element-scoped locking is leaky. Expected {}, got {}",
            4 * ops_per_thread,
            total_sum
        );

        println!("Element RW Race Hammer passed. Total sum: {}", total_sum);
    }
}
