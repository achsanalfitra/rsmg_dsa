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

    pub(crate) fn update<F>(&self, mut f: F)
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
}

unsafe impl<T: Copy + Send> Sync for AtomicHandle<T> {}
unsafe impl<T: Copy + Send> Send for AtomicHandle<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_atomic_integrity_1000_threads() {
        let size: usize = 0;
        let handle = Arc::new(AtomicHandle::new(size));
        let mut workers = Vec::with_capacity(1000);

        for _ in 0..1000 {
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

        assert_eq!(final_val, 1000);
        assert!(stamp >= 1000, "stamp incorrect");
    }
}
