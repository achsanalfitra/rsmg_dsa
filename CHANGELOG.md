## 2026-01-10 — alpha1.1
### What’s new

* ContiguousArray
  * Introduced a new concurrent, index-addressable, growable structure
  * Designed for scoped inspection and mutation without a global lock
  * Covered by concurrency stress tests
  * Passes Miri
  * test_inspect_during_pop_of_same_index was particularly painful (more testing is welcome)
* LinkedStack
 * Now fully passes Miri
 * No leaks detected
 * Safe to use within the semantics exercised by the tests

* Notes
 * Unsafe code paths are Miri-verified
 * APIs are structured to prevent misuse (no manual lock/unlock)

* HOTPATCH
 * Remove Copy trait bound
