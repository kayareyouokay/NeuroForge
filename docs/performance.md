# Performance Notes

The current runtime is eager and CPU-bound.  That makes it easy to debug, but performance depends heavily on whether an operation stays in vectorized NumPy code.

## Current Hot Paths

- Matmul and reductions call NumPy directly and are reasonably efficient for CPU experiments.
- Convolution forward uses `tensordot` over sliding windows; backward is explicit and correct, but not yet fast.
- Attention uses batched matmul and should be acceptable for small sequence lengths.
- RNN/LSTM unroll in Python.  This is useful for clarity and gradient debugging, but not ideal for long sequences.

## Memory

Autograd retains parent tensors and backward closures until the graph is released.  Large sequence models will therefore hold activations for every layer and timestep.  Obvious future work:

- activation checkpointing
- graph lifetime profiling
- in-place optimizer buffers with dtype policies
- mixed precision once backend/device semantics exist

## Benchmarking

Run:

```bash
python benchmarks/bench_autograd.py --batch 256 --width 256 --steps 100
```

TODO: add benchmark baselines for convolution backward, attention, and recurrent unrolls.  The benchmark harness is intentionally boring so regressions are easy to spot in CI later.
