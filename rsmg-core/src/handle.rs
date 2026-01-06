use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, Ordering};

static BEGIN_STAMP: u32 = 2;
static WRITING_STATE: u32 = 1;

enum STATE {
    Success,
    Stale,
    Busy,
}

pub(crate) struct AtomicHandle<T> {
    pub(crate) inner: UnsafeCell<T>,
    pub(crate) stamp: AtomicU32,
}

impl<T> AtomicHandle<T> {
    pub(crate) fn new(inner: T) -> Self {
        Self {
            inner: UnsafeCell::new(inner),
            stamp: AtomicU32::new(BEGIN_STAMP),
        }
    }

    pub(crate) fn get(&self) -> (T, u32)
    where
        T: Copy,
    {
        loop {
            let stamp = self.stamp.load(Ordering::Acquire);

            if stamp == WRITING_STATE {
                core::hint::spin_loop();
                continue;
            }

            let inner = unsafe { *self.inner.get() };

            let stamp_after = self.stamp.load(Ordering::Acquire);

            if stamp == stamp_after {
                return (inner, stamp);
            }
        }
    }

    pub(crate) fn update<F>(&self, mut f: F) -> T
    where
        F: FnMut(T) -> T,
        T: Copy,
    {
        let (mut current_val, mut current_stamp) = self.get();

        loop {
            let next_val = f(current_val);

            match self.swap(next_val, current_stamp) {
                STATE::Success => break,

                STATE::Busy => {
                    core::hint::spin_loop();
                    continue;
                }

                STATE::Stale => {
                    core::hint::spin_loop();
                    let (v, s) = self.get();
                    current_val = v;
                    current_stamp = s;
                }
            }
        }

        current_val
    }

    pub(crate) fn update_exclusive<F>(&self, mut f: F) -> T
    where
        F: FnMut(T) -> T,
        T: Copy,
    {
        let (mut current_val, mut current_stamp) = self.get();

        loop {
            match self.write(current_val, current_stamp, &mut f) {
                (data, STATE::Success) => return data,

                (_, STATE::Busy) => {
                    core::hint::spin_loop();
                    continue;
                }

                (_, STATE::Stale) => {
                    core::hint::spin_loop();
                    let (v, s) = self.get();
                    current_val = v;
                    current_stamp = s;
                }
            }
        }
    }

    fn swap(&self, inner: T, stamp: u32) -> STATE {
        if self.stamp.load(Ordering::Acquire) == WRITING_STATE {
            return STATE::Busy;
        }

        let mut new_stamp = stamp.wrapping_add(1);
        if new_stamp == WRITING_STATE {
            new_stamp = new_stamp.wrapping_add(1);
        }

        match self
            .stamp
            .compare_exchange(stamp, WRITING_STATE, Ordering::AcqRel, Ordering::Relaxed)
        {
            Ok(_) => {
                unsafe { *self.inner.get() = inner };
            }
            _ => return STATE::Stale,
        };

        self.stamp.store(new_stamp, Ordering::Release);

        STATE::Success
    }

    fn write<F>(&self, inner: T, stamp: u32, mut f: F) -> (T, STATE)
    where
        F: FnMut(T) -> T,
    {
        if self.stamp.load(Ordering::Acquire) == WRITING_STATE {
            return (inner, STATE::Busy);
        }

        let mut new_stamp = stamp.wrapping_add(1);
        if new_stamp == WRITING_STATE {
            new_stamp = new_stamp.wrapping_add(1);
        }

        match self
            .stamp
            .compare_exchange(stamp, WRITING_STATE, Ordering::AcqRel, Ordering::Relaxed)
        {
            Ok(_) => {
                let new_inner = f(inner);
                self.stamp.store(new_stamp, Ordering::Release);
                return (new_inner, STATE::Success);
            }
            _ => return (inner, STATE::Stale),
        };
    }
}

unsafe impl<T: Copy + Send> Sync for AtomicHandle<T> {}
unsafe impl<T: Copy + Send> Send for AtomicHandle<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_atomic_integrity_targeted_threads() {
        static TARGET: usize = 1000;
        let size: usize = 0;
        let handle = Arc::new(AtomicHandle::new(size));
        let mut workers = Vec::with_capacity(TARGET);

        for _ in 0..TARGET {
            let h = Arc::clone(&handle);
            workers.push(thread::spawn(move || {
                h.update(|val| {
                    let new_val = val + 1;
                    new_val
                });
            }));
        }

        for worker in workers {
            worker.join().expect("thread panicked");
        }

        let (final_val, stamp) = handle.get();

        assert_eq!(final_val, TARGET);
        assert!(stamp >= TARGET as u32, "stamp incorrect");
    }

    #[derive(Clone)]
    pub(crate) struct TestStruct {
        pub(crate) id: i32,
        pub(crate) name: String,
    }

    impl TestStruct {
        fn new(id: i32, name: String) -> Self {
            Self { id: id, name: name }
        }
    }

    impl Drop for TestStruct {
        fn drop(&mut self) {
            println!("memory freed for name: {}", self.name);
        }
    }

    #[test]
    fn test_pointer_manipulation() {
        let test_object = TestStruct::new(1, "hello".to_string());
        let ptr = Box::into_raw(Box::new(test_object));

        let handle = AtomicHandle::new(ptr);

        // the writer is forced to clone.
        // This elevates the burden from AtomicHandle
        // into the primitives that use this
        let old_ptr = handle.update(|ptr| {
            let mut safe_ptr = unsafe { (*ptr).clone() };
            safe_ptr.name = "world".to_string();
            Box::into_raw(Box::new(safe_ptr))
        });

        let (current_ptr, _stamp) = handle.get();

        assert_eq!(unsafe { &current_ptr.as_ref().unwrap().name }, "world");

        unsafe {
            let _ = Box::from_raw(old_ptr);
        } // free() dance
    }

    #[test]
    fn test_arc_manipulation() {
        let test_object = Arc::into_raw(Arc::new(TestStruct::new(1, "hello".to_string())));
        let handle = AtomicHandle::new(test_object);

        let old_ptr = handle.update(|old_arc| {
            let mut new_obj = unsafe { (*old_arc).clone() };
            new_obj.name = "world".to_string();

            Arc::into_raw(Arc::new(new_obj))
        });

        let (current_arc, _) = handle.get();
        assert_eq!(unsafe { &current_arc.as_ref().unwrap().name }, "world");

        unsafe {
            Arc::decrement_strong_count(old_ptr);
        }
    }
}
