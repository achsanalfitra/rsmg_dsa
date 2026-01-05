# Rust SMGEN Data Structure and Algorithm

This provides a lean library that uses a lock-free and MPMC-compatible via CAS.
The primitives uses the `AtomicHandle` to handle the swap under the hood, while
the primtive itself can be imported as one struct, without runtime or backend dependencies.

Unstable: this is currently unstable, since there are barely any features yet.
Contributors are welcomed.
