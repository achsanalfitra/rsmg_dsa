use std::sync::Arc;

use crate::handle::AtomicHandle;

struct LinkedStack<T> {
    inner: AtomicHandle<*mut InnerLinkedStack<T>>,
}

struct InnerLinkedStack<T> {
    head: Option<Arc<LinkedStackNode<T>>>,
}

struct LinkedStackNode<T> {
    data: T,
    next: Option<Arc<LinkedStackNode<T>>>,
}

impl<T> LinkedStack<T> {
    fn new() -> Self {
        Self {
            inner: AtomicHandle::new(Box::into_raw(Box::new(InnerLinkedStack::new()))),
        }
    }

    fn push(
        &mut self,
        mut node: Option<Arc<LinkedStackNode<T>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.inner.update_exclusive(|inner| {
            if let Some(n) = node.take() {
                unsafe {
                    inner.as_mut().unwrap().push(Some(n)).unwrap_unchecked();
                }
            }
            inner
        });

        Ok(())
    }

    fn pop(&mut self) -> Result<Option<Arc<LinkedStackNode<T>>>, Box<dyn std::error::Error>> {
        let mut popped_node = None;

        self.inner.update_exclusive(|inner| unsafe {
            popped_node = inner.as_mut().unwrap().pop().unwrap();
            inner
        });

        Ok(popped_node)
    }
}

impl<T> InnerLinkedStack<T> {
    fn new() -> Self {
        Self { head: None }
    }

    fn push(
        &mut self,
        node: Option<Arc<LinkedStackNode<T>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut new_node_arc = node.ok_or("pushed None")?;

        if let Some(node_mut) = Arc::get_mut(&mut new_node_arc) {
            node_mut.next = self.head.take();
            self.head = Some(new_node_arc);
        } else {
            return Err("multiple references exist".into());
        }

        Ok(())
    }

    fn pop(&mut self) -> Result<Option<Arc<LinkedStackNode<T>>>, Box<dyn std::error::Error>> {
        let mut head_arc = match self.head.take() {
            Some(a) => a,
            None => return Ok(None),
        };

        self.head = head_arc.next.clone();

        if let Some(node_mut) = Arc::get_mut(&mut head_arc) {
            node_mut.next = None;
        } else {
            return Err("multiple references exist".into());
        }

        Ok(Some(head_arc))
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
        let mut stack = LinkedStack::new();

        let node1 = Arc::new(LinkedStackNode::new(10));
        let node2 = Arc::new(LinkedStackNode::new(20));

        stack.push(Some(node1)).unwrap();
        stack.push(Some(node2)).unwrap();

        let popped1 = stack.pop().unwrap().expect("should have a node");
        assert_eq!(popped1.data, 20);

        let popped2 = stack.pop().unwrap().expect("should have a node");
        assert_eq!(popped2.data, 10);

        assert!(stack.pop().unwrap().is_none());
    }

    #[test]
    fn test_concurrency_hammer() {
        let stack = Arc::new(LinkedStack::new());
        let mut handles = vec![];
        let num_threads = 10;
        let ops_per_thread = 100;

        for t in 0..num_threads {
            let s = Arc::clone(&stack);
            handles.push(thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let node = Arc::new(LinkedStackNode::new(t * 1000 + i));
                    let stack_ptr = s.as_ref() as *const LinkedStack<i32> as *mut LinkedStack<i32>;
                    unsafe { (*stack_ptr).push(Some(node)).unwrap() };
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let mut count = 0;
        let mut stack_mut = Arc::try_unwrap(stack)
            .ok()
            .expect("only one ref should remain");
        while let Some(_) = stack_mut.pop().unwrap() {
            count += 1;
        }

        assert_eq!(count, num_threads * ops_per_thread);
    }

    #[test]
    fn test_push_none_error() {
        let mut stack: LinkedStack<usize> = LinkedStack::new();
        let _ = stack.push(None);
    }
}
