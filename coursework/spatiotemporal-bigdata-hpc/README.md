# Spatiotemporal Big Data HPC Labs

[English](README.md) | [中文](README.zh-CN.md)

<p align="center">
  <a href="figures/lab1_float_crash.gif"><img src="figures/lab1_float_crash.gif" alt="Single precision accumulator failure" width="49%" /></a>
  <a href="figures/lab4_warp_morph.gif"><img src="figures/lab4_warp_morph.gif" alt="Affine warp with bilinear resampling" width="49%" /></a>
</p>
<p align="center">
  <a href="figures/lab2_fit_stream.gif"><img src="figures/lab2_fit_stream.gif" alt="Streaming least squares fit" width="49%" /></a>
  <a href="figures/lab3_mpi_partition.gif"><img src="figures/lab3_mpi_partition.gif" alt="MPI interval split and reduction" width="49%" /></a>
</p>
<p align="center"><b>Four animated demos: float crash → streaming fit → MPI split-reduce → affine warp</b></p>

**Four labs on high performance processing: numerical integration with a float accumulation failure study, matrix algebra from scratch with least squares fitting, distributed memory MPI versions, and shared memory OpenMP affine image warping. All programs are C++17, built with g++ and OpenMPI and measured on the same machine.**

The labs build up one stack: the integral lab measures rectangle and trapezoid summation from 10³ to 10⁹ intervals and uncovers the single precision accumulator collapse with a full mechanism analysis; the matrix lab implements the four basic matrix operations and drives a least squares fit through them; the MPI lab turns both the integral and the fit into distributed versions with process scaling tables; the OpenMP lab fits a 6 parameter affine transform from four corner pairs and resamples a 12001 × 11001 image with inverse mapping and bilinear interpolation.

**Status.** Coursework complete. Reports and defense material live in the course workspace; this repository keeps the core sources, the full figure set and the animated demos.

## My Role

Bohan Yu completed all four labs end to end: the C++ implementations from scratch, including the four matrix operations, the Gauss-Jordan inverse, the Kahan compensated summation, the diagnostic programs that pin down the float accumulation failure, the MPI partitioned versions of the integral and the least squares fit, and the OpenMP image warping with the least squares affine fit. The measurements, figures, animated demos and documentation are part of the same effort.

## Labs

| # | Lab | Method | Key result |
|---|---|---|---|
| 1 | Integral computation | Rectangle and trapezoid summation, 12 sizes from 10³ to 10⁹, float / double, Kahan compensation | T(n) ∝ n, 8.5 μs → 7.92 s; trapezoid error 4.03e-14; the float accumulator collapses to 1.0 with error 5.4e-2, freezes at 2²⁴; Kahan restores the float error to 1.5e-9 |
| 2 | Matrix algebra + least squares | Four matrix operations from scratch, Gauss-Jordan inverse, normal equations | k = 2.9997402, b = 1.9997501; 0.35 μs at 10 points, 7.6 ns per point at 10³, 25.7 ms at 10⁶ |
| 3 | MPI distributed memory | Block partition with MPI_Reduce, 1 / 2 / 4 / 8 processes | Integral 5.43 → 1.05 s, speedup 5.16×; fit 3.26 → 0.62 ms, speedup 5.24×; identical results at every process count |
| 4 | OpenMP shared memory | 8-equation least squares affine fit, inverse mapping with bilinear resampling | 132 M output pixels, 0.510 s serial → 0.110 s at 8 threads, speedup 4.64× |

## Results Preview

### 1 Integral computation

<p align="center">
  <a href="figures/lab1_time_vs_intervals.png"><img src="figures/lab1_time_vs_intervals.png" alt="Time versus interval count" width="78%" /></a>
</p>

Time grows in proportion to the interval count across six orders of magnitude. The top decade bends upward as sustained runs lose CPU frequency.

<p align="center">
  <a href="figures/lab1_error_vs_intervals.png"><img src="figures/lab1_error_vs_intervals.png" alt="Error versus interval count" width="78%" /></a>
</p>

Rectangle error follows O(h) and trapezoid error follows O(h²); the double curves stop improving near machine precision, and the float curves diverge after n = 10⁴.

<p align="center">
  <a href="figures/lab1_float_accumulator.png"><img src="figures/lab1_float_accumulator.png" alt="Float accumulator mechanism" width="78%" /></a>
</p>

The accumulator becomes an integer counter once ulp reaches 0.5, pins the result at 1.0 (error 5.4e-2), then freezes at 2²⁴ and decays as 2²⁴/n. The diagnostic counts of ineffective additions match n − 2²⁴ exactly.

### 2 Matrix algebra with least squares fitting

<p align="center">
  <a href="figures/lab2_time_vs_points.png"><img src="figures/lab2_time_vs_points.png" alt="Time versus point count" width="72%" /></a>
</p>

The fit runs in a single pass through the normal equations. The three scales land in the fixed overhead regime, the cache resident regime and the memory bandwidth regime.

### 3 MPI distributed memory programming

<p align="center">
  <a href="figures/lab3_time_vs_processes.png"><img src="figures/lab3_time_vs_processes.png" alt="Time versus process count" width="49%" /></a>
  <a href="figures/lab3_speedup_vs_processes.png"><img src="figures/lab3_speedup_vs_processes.png" alt="Speedup versus process count" width="49%" /></a>
</p>

Both programs scale close to linear up to 4 processes. The fit keeps its communication at five doubles per reduction, so its parallel overhead tracks the process count rather than the point count.

### 4 OpenMP shared memory programming

<p align="center">
  <a href="figures/lab4_source_image.png"><img src="figures/lab4_source_image.png" alt="Source image" width="44%" /></a>
  <a href="figures/lab4_warped_image.png"><img src="figures/lab4_warped_image.png" alt="Warped image" width="44%" /></a>
</p>

The affine model rotates and shears the diagonal stripe pattern; pixels outside the source coverage render black, which matches the intended canvas bounds.

## Run

```bash
# Lab 1  integral, diagnostics
g++ -O2 -std=c++17 lab1-integral/integral_lab.cpp -o integral_lab -lm
./integral_lab --full
g++ -O2 -std=c++17 lab1-integral/diag4.cpp  -o diag4  -lm && ./diag4
g++ -O2 -std=c++17 lab1-integral/diag12.cpp -o diag12 -lm && ./diag12

# Lab 2  matrix operations and least squares
g++ -O2 -std=c++17 lab2-matrix-lstsq/lab2_matrix_ls.cpp -o lab2 -lm
taskset -c 2 ./lab2

# Lab 3  MPI versions
mpicxx -O2 -std=c++17 lab3-mpi/mpi_integral.cpp -o mpi_integral -lm
mpicxx -O2 -std=c++17 lab3-mpi/mpi_lsfit.cpp    -o mpi_lsfit    -lm
for p in 1 2 4 8; do mpirun -np $p ./mpi_integral; done
for p in 1 2 4 8; do mpirun -np $p ./mpi_lsfit;    done

# Lab 4  OpenMP affine warp
g++ -O2 -std=c++17 -fopenmp lab4-openmp/warp_affine.cpp -o warp -lm
./warp
```

## Layout

```
spatiotemporal-bigdata-hpc/
├─ lab1-integral/      integral_lab.cpp, diag4.cpp, diag12.cpp, diag13.cpp
├─ lab2-matrix-lstsq/  lab2_matrix_ls.cpp
├─ lab3-mpi/           mpi_integral.cpp, mpi_lsfit.cpp
├─ lab4-openmp/        warp_affine.cpp
└─ figures/            charts, result images and four animated demos
```

## Notes

The measurements ran on a 12th Gen Intel Core i7-12700 under WSL2 Ubuntu 22.04 with g++ 11.4 and OpenMPI 4.1. Timing tables in the course reports take the best of repeated rounds; the machine carried background load, which the reports discuss in the analysis sections.
