#RSMG-CORE

This library contains the core data structures
and atomic pointer utilities used by the RSMG framework.
The implementation primarily covers lock-free linked
structures and atomic handles required for shared state management in
performance-sensitive applications.

Current features:
* Public facing API
  * LinkedStack [passed miri]
  * ContiguousArray [passed miri] (but, test_inspect_during_pop_of_same_index
  has been really difficult to deal with. Further testing is appreciated)

* Internal facing API
  * AtomicHandle (unstable functions: update())

Future work:
  * The ContiguousArray only supports push/pop semantics for now.
  However, it's already index-addressable. Future work includes,
    1. Add remove/insert semantics
    2. Add read-only accessor semantics
