## Performance Results

| Configuration | bfly_G_10.mtx | D6-6.mtx | dictionary28.mtx | Ga3As3H12.mtx | pkustk14.mtx | roadNet-CA.mtx |
|--------------|---------------|----------|------------------|---------------|--------------|----------------|
| **Sequential** | 0.2397 ms<br>(1.64 GFLOP/s, 14.8 GB/s) | 0.1828 ms<br>(1.61 GFLOP/s, 15.1 GB/s) | 0.3240 ms<br>(1.10 GFLOP/s, 9.8 GB/s) | 19.9863 ms<br>(0.60 GFLOP/s, 4.8 GB/s) | 49.3411 ms<br>(0.60 GFLOP/s, 4.8 GB/s) | 9.6308 ms<br>(1.15 GFLOP/s, 10.8 GB/s) |
| **MPI** (N=8, n=8) | 0.0254 ms<br>(1.93 GFLOP/s, 17.4 GB/s) | 0.0176 ms<br>(2.09 GFLOP/s, 19.5 GB/s) | 0.0316 ms<br>(1.41 GFLOP/s, 11.8 GB/s) | 1.4859 ms<br>(1.00 GFLOP/s, 8.1 GB/s) | 4.1089 ms<br>(0.90 GFLOP/s, 7.3 GB/s) | 0.7493 ms<br>(1.85 GFLOP/s, 17.4 GB/s) |
| **OpenMP** (N=1, n=8, threads=16) | 0.1483 ms<br>(2.65 GFLOP/s, 23.9 GB/s) | 0.1094 ms<br>(2.69 GFLOP/s, 25.2 GB/s) | 0.1381 ms<br>(2.58 GFLOP/s, 22.9 GB/s) | 4.6648 ms<br>(2.56 GFLOP/s, 20.6 GB/s) | 11.4789 ms<br>(2.59 GFLOP/s, 20.8 GB/s) | 4.0408 ms<br>(2.74 GFLOP/s, 25.8 GB/s) |
| **Hybrid** (N=2, n=8, threads=2) | 0.1742 ms<br>(0.28 GFLOP/s, 2.5 GB/s) | 0.1247 ms<br>(0.29 GFLOP/s, 2.7 GB/s) | 0.1489 ms<br>(0.30 GFLOP/s, 2.5 GB/s) | 4.9985 ms<br>(0.30 GFLOP/s, 2.4 GB/s) | 12.3509 ms<br>(0.30 GFLOP/s, 2.4 GB/s) | 4.7336 ms<br>(0.29 GFLOP/s, 2.8 GB/s) |
| **Hybrid** (N=4, n=8, threads=2) | 0.1544 ms<br>(0.32 GFLOP/s, 2.9 GB/s) | 0.1074 ms<br>(0.34 GFLOP/s, 3.2 GB/s) | 0.1293 ms<br>(0.34 GFLOP/s, 2.9 GB/s) | 4.3260 ms<br>(0.35 GFLOP/s, 2.8 GB/s) | 10.6512 ms<br>(0.35 GFLOP/s, 2.8 GB/s) | 4.0787 ms<br>(0.34 GFLOP/s, 3.2 GB/s) |
| **Hybrid** (N=8, n=8, threads=2) | 0.2337 ms<br>(0.21 GFLOP/s, 1.9 GB/s) | 0.1703 ms<br>(0.22 GFLOP/s, 2.0 GB/s) | 0.2261 ms<br>(0.20 GFLOP/s, 1.7 GB/s) | 7.4550 ms<br>(0.20 GFLOP/s, 1.6 GB/s) | 18.5926 ms<br>(0.20 GFLOP/s, 1.6 GB/s) | 5.9112 ms<br>(0.23 GFLOP/s, 2.2 GB/s) |

**Note:** N = number of nodes, n = number of processes, threads = OpenMP threads per process
