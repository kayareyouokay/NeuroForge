# Roadmap

## 0.2

- Vectorized convolution backward using im2col/col2im
- Gradient clipping and optimizer parameter groups
- Module buffers for BatchNorm running statistics
- More complete checkpoint schema with version stamping
- Better error messages for shape mismatches

## 0.3

- Device object and backend registry
- Experimental OpenCL or CUDA backend spike
- Profiling hooks for op timing and retained activation memory
- Optional activation checkpointing for Transformer stacks

## 0.4

- Mixed precision policy
- Data loading utilities
- More examples with real datasets
- Exportable training logs

## Later

- Tiny compiler/runtime experiment for fused elementwise operations
- Distributed training sketch
- Model zoo kept deliberately small and test-backed
