## Performance Results

| Configuration | bfly_G_10.mtx | D6-6.mtx | dictionary28.mtx | Ga3As3H12.mtx | pkustk14.mtx | roadNet-CA.mtx |
|--------------|---------------|----------|------------------|---------------|--------------|----------------|
| **CUDA** | 0.0085 ms<br>(46.12 GFLOP/s, 415.0 GB/s) | 0.0088 ms<br>(33.48 GFLOP/s, 314.2 GB/s) | 0.0104 ms<br>(34.10 GFLOP/s, 302.9 GB/s) | 1.6525 ms<br>(7.23 GFLOP/s, 58.1 GB/s) | 2.6145 ms<br>(11.35 GFLOP/s, 91.3 GB/s) | 0.3667 ms<br>(30.18 GFLOP/s, 284.3 GB/s) |
| **CUDA-MPI** (N=2, n=2) | 0.0055 ms<br>(35.76 GFLOP/s, 321.8 GB/s) | 0.0054 ms<br>(27.11 GFLOP/s, 252.1 GB/s) | 0.0083 ms<br>(21.47 GFLOP/s, 184.6 GB/s) | 1.3018 ms<br>(4.59 GFLOP/s, 36.9 GB/s) | 2.3609 ms<br>(6.28 GFLOP/s, 50.5 GB/s) | 0.1435 ms<br>(38.57 GFLOP/s, 362.9 GB/s) |
| **CUDA-MPI** (N=4, n=4) | 0.0037 ms<br>(26.41 GFLOP/s, 237.6 GB/s) | 0.0038 ms<br>(19.27 GFLOP/s, 179.2 GB/s) | 0.0063 ms<br>(14.14 GFLOP/s, 119.5 GB/s) | 0.6637 ms<br>(4.50 GFLOP/s, 36.2 GB/s) | 1.1911 ms<br>(6.23 GFLOP/s, 50.0 GB/s) | 0.0755 ms<br>(36.66 GFLOP/s, 345.0 GB/s) |

**Note:** N = number of nodes, n = number of processes
