use crate::handle::AtomicHandle;

struct LinkedStack<T> {
    inner: AtomicHandle<*mut InnerLinkedStack<T>>,
}

struct InnerLinkedStack<T> {
    head: Option<*mut LinkedStackNode<T>>,
}

struct LinkedStackNode<T> {
    data: T,
    next: Option<*mut LinkedStackNode<T>>,
}

impl<T> LinkedStack<T> {
    fn new() -> Self {
        Self {
            inner: AtomicHandle::new(Box::into_raw(Box::new(InnerLinkedStack::new()))),
        }
    }

    fn pop(&self) -> Result<Option<T>, Box<dyn std::error::Error>> {
        let mut popped_node = None;

        self.inner.update_exclusive(|inner| unsafe {
            popped_node = inner.as_mut().unwrap().pop().unwrap();
            inner
        });

        Ok(popped_node)
    }

    fn push(&self, node: LinkedStackNode<T>) -> Result<(), Box<dyn std::error::Error>> {
        let node_ptr = Box::into_raw(Box::new(node));
        if node_ptr.is_null() {
            return Err("pushed null pointer".into());
        }

        let mut result = Ok(());

        self.inner.update_exclusive(|inner_ptr| {
            unsafe {
                let inner = inner_ptr.as_mut().unwrap();
                result = inner.push(Some(node_ptr));
            }
            inner_ptr
        });

        result
    }
}

impl<T> InnerLinkedStack<T> {
    fn new() -> Self {
        Self { head: None }
    }

    fn push(
        &mut self,
        node: Option<*mut LinkedStackNode<T>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(current_node) = node {
            unsafe {
                (*current_node).next = self.head;
            }
            self.head = Some(current_node);
        }

        Ok(())
    }

    fn pop(&mut self) -> Result<Option<T>, Box<dyn std::error::Error>> {
        if let Some(node_mut) = self.head {
            unsafe {
                self.head = (*node_mut).next;

                let boxed_node = Box::from_raw(node_mut);

                return Ok(Some(boxed_node.data));
            }
        }

        Ok(None)
    }
}

impl<T> Drop for InnerLinkedStack<T> {
    fn drop(&mut self) {
        while let Ok(Some(_)) = self.pop() {}
    }
}

impl<T> LinkedStackNode<T> {
    fn new(data: T) -> Self {
        Self {
            data: data,
            next: None,
        }
    }
}

unsafe impl<T: Copy + Send> Sync for LinkedStack<T> {}
unsafe impl<T: Copy + Send> Send for LinkedStack<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_push_pop() {
        let stack = LinkedStack::new();

        let node1 = LinkedStackNode::new(10);
        let node2 = LinkedStackNode::new(20);

        stack.push(node1).unwrap();
        stack.push(node2).unwrap();

        let popped1 = stack.pop().unwrap().expect("should have a node");
        assert_eq!(popped1, 20);

        let popped2 = stack.pop().unwrap().expect("should have a node");
        assert_eq!(popped2, 10);

        assert!(stack.pop().unwrap().is_none());
    }

    #[test]
    fn test_concurrency_hammer() {
        let stack = Arc::new(LinkedStack::new());
        let mut handles = vec![];
        let num_threads = 10;
        let ops_per_thread = 1000;
        for t in 0..num_threads {
            let s = Arc::clone(&stack);
            handles.push(thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let data = t * 1000 + i;
                    let node = LinkedStackNode::new(data);
                    s.push(node).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let mut count = 0;
        let stack_final = Arc::try_unwrap(stack).ok().expect("Arc leak detected");

        let stack_inner = stack_final;
        while let Ok(Some(_)) = stack_inner.pop() {
            count += 1;
        }

        assert_eq!(count, num_threads * ops_per_thread);
    }

    #[test]
    fn test_concurrency_zero_sum_hammer() {
        let stack = Arc::new(LinkedStack::new());
        let mut handles = vec![];
        let num_threads = 10;
        let ops_per_thread = 1000; // Total 10,000 operations

        for t in 0..num_threads {
            let s = Arc::clone(&stack);
            handles.push(thread::spawn(move || {
                for i in 0..ops_per_thread {
                    // 50% Push, 50% Pop
                    if i % 2 == 0 {
                        let data = t * 1000 + i;
                        let node = LinkedStackNode::new(data);
                        s.push(node).expect("push should never fail");
                    } else {
                        let _ = s.pop().expect("pop should never fail");
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // After all threads finish, the stack SHOULD be empty
        // because each thread did 500 pushes and 500 pops.
        let mut final_count = 0;
        let stack_final = Arc::try_unwrap(stack).ok().expect("Arc leak detected");

        while let Ok(Some(_)) = stack_final.pop() {
            final_count += 1;
        }

        assert_eq!(final_count, 0, "stack was not empty!");
    }
}
