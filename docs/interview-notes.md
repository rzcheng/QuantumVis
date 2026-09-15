# Concepts to work through personally

These are exercises to derive and inspect, not a script to memorize. GPU source
exists, but actual NVIDIA correctness and performance are still unverified.

1. **Why pairs?** Write a three-qubit state as a sum over the other two bits.
   A gate on one qubit leaves those other bits fixed and mixes just its zero/one
   amplitudes. Derive `i1 = i0 XOR (1 << target)` for a target-zero `i0`.
2. **Why target-dependent access?** Enumerate indices for targets 0 and 2 in an
   eight-amplitude vector. Pair separation is `2**target`. Now enumerate successive
   pair numbers: low targets give strided streams; high targets give longer
   contiguous runs. Logical accesses alone do not prove achieved GPU bandwidth.
3. **Why one owner?** Insert a zero target bit into a compact pair number. Derive
   the inverse mapping and show all pairs are disjoint and cover the state.
   Assigning both members separate workers would create duplicate writes/races.
4. **Why in place?** Both old complex amplitudes must be loaded before either is
   overwritten. Independent pairs make this safe without atomics. Explain how
   out-of-place storage changes allocation, footprint, and lifetime, and why a
   different operation might require it.
5. **Why precision differences?** Trace input rounding, coefficient rounding, and
   arithmetic rounding from complex128 to split float32. Cancellation can make a
   tiny amplitude's relative error large. Inspect amplitude, relative L2, and norm
   errors separately; never repair a failed comparison with silent normalization.
6. **Why norm is insufficient?** A wrong permutation or phase gate can remain
   unitary and preserve norm exactly. Construct an example that passes norm checks
   while giving wrong amplitudes, then one with unchanged probabilities but wrong
   phase. This motivates analytical and independent differential checks.
7. **What does synchronization do?** GPU launches enqueue work; a host timer can
   otherwise stop before execution ends. Explain device synchronization versus
   stream ordering, and why blocking downloads make the current public result ready.
   Future device timing must distinguish warmup/compilation from steady-state work.
8. **Which latency?** Draw the boundaries around raw kernel execution, launch and
   gate wrapper overhead, transfers, and full circuit execution. Today's CPU JSON
   times complete one-gate `run()` calls, including validation and copies. A GPU
   kernel-only time would not be a fair direct comparison to that metric.

Before claiming an improvement, name the measured workload, precision, hardware,
software versions, baseline, metric boundary, raw artifact, and correctness evidence.
After first GPU correctness, complete controlled-gate coverage and establish fair
GPU/CPU measurements before selecting an optimization from profiler evidence.
