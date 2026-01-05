#[derive(Debug, Clone, Eq, PartialEq)]
struct ListNode {
    pub next: Option<Box<ListNode>>,
    pub val: i32,
}

impl ListNode {
    pub fn new(val: i32) -> Self {
        Self {
            next: None,
            val: val,
        }
    }

    pub fn assign(&mut self, next: Box<ListNode>) {
        self.next = Some(next);
    }

    pub fn from_vec(vec: Vec<i32>) -> Option<Box<Self>> {
        let mut head = None;

        for &val in vec.iter().rev() {
            let mut current = ListNode::new(val);
            current.next = head;
            head = Some(Box::new(current));
        }
        head
    }

    pub fn into_vec(&self, mut head: Option<Box<ListNode>>) -> Vec<i32> {
        let mut vec: Vec<i32> = Vec::new();

        while let Some(mut curr) = head {
            vec.push(curr.val);
            let next = curr.next.take();
            head = next;
        }
        vec
    }

    pub fn reverse(&self, mut head: Option<Box<ListNode>>) -> Option<Box<Self>> {
        let mut rev_head = None;

        while let Some(mut curr) = head {
            let next = curr.next.take();
            curr.next = rev_head;
            rev_head = Some(curr);
            head = next;
        }

        rev_head
    }
}
