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
}

impl ContiguousArrayMetadata {
    fn new() -> Self {
        Self {
            length: DEFAULT_LENGTH,
            capacity: DEFAULT_CAPACITY,
            locker_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
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
}
