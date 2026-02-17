# Benchmark Code Collection — Conservative Finite-Volume Transport (C++/CUDA, MATLAB, Julia)

This repository is a **multi-language benchmark suite** that implements the *same* conservative finite-volume transport simulation
in several backends (serial CPU, multi-core CPU, GPU) and measures **time-per-step vs grid size**.

It is intended to answer questions like: *“How fast can this specific transport + spline + convolution workload run in C++/OpenMP vs CUDA vs MATLAB vs Julia?”*

---

## What is being simulated?

All implementations evolve a 2D field `phi(z, r, t)` on an `N × N` grid (discrete axial coordinate `z` and radial coordinate `r`).

Each time step (conceptually) does:

1. **Radial integration** to obtain a 1D profile  
   `psi(z) = Σ_r phi(z, r)` (discrete sum over the radial index).

2. **Spline interpolation** of `psi` to a refined 1D grid of length  
   `M = floor(N * (65536/N) + 1)` → **`M ≈ 65537` for all `N`**, by construction.

3. **1D convolution / correlation** of the interpolated profile with a fixed 31-tap kernel (`kernelInput.dat`) to obtain `I(z)`.

4. **Compute axial advection velocity** (per cell) from a static geometry/gradient term plus the interaction signal `I(z)`:
   `v_z(z,r) = -α * (R(r) + P(z) - P0) - β * I(z)`  
   (details and region logic in the code; e.g. “wing” vs “center” uses different `P(z)` profiles).

5. **Conservative finite-volume update** using:
   - **Upwind** advection in `z`
   - optional diffusion in `z` and `r`
   - **zero-flux boundaries** (sealed box / mass-conserving)
   - clamp `phi < 0` to `0`

### Complexity (why this is a good benchmark)
The dominant work scales like **O(N²)** (building velocities + fluxes + updating `phi`). The spline+convolution part is ~constant-cost per step because `M` is ~constant.

---

## Repository layout

- `cpp_serial/` — C++ single-thread implementation (pins to core 0, FTZ/DAZ enabled)
- `cpp_parallel/` — C++ OpenMP implementation
- `cpp_cuda/` — C++/CUDA implementation (unity build via `benchmark_main.cu`)
- `matlab_serial/` — MATLAB CPU implementation
- `matlab_vectorized/` — MATLAB CPU “optimized/vectorized” implementation
- `matlab_gpu_array/` — MATLAB `gpuArray` implementation (Parallel Computing Toolbox)
- `julia_cpu_serial/` — Julia CPU serial implementation
- `julia_cpu_parallel/` — Julia CPU threaded implementation
- `julia_cuda/` — Julia CUDA implementation (CUDA.jl)

Top-level helpers / data:

- `kernelInput.dat` — **required** kernel file (at least 31 values; the first 31 are used)
- `*_cpu.txt`, `*_cuda.txt`, … — example benchmark result tables (`N`, `avg_ms`, `std_ms`)
- `benchmark_plotter.m` — plots the `*.txt` benchmark tables

---

## Common command-line / function signature

Most backends use the same parameter order (first argument required):

| Position | Name | Meaning | Typical default |
|---:|---|---|---|
| 1 | `N` | grid size (`N × N`) | (required) |
| 2 | `ISF` | interaction scale factor applied to kernel | `1.0` |
| 3 | `PSI` | average volume fraction target used for initialization/normalization | `0.02` |
| 4 | `IT` | time step `dt` | `0.005` (benchmarks often use `0.001`) |
| 5 | `T` | total simulated time | `1200.0` (benchmarks often scale with `1/N`) |
| 6 | `NO` | output interval (steps) | `1000` (benchmarks use large values to avoid I/O) |
| 7 | `D` | diffusion coefficient | `0.0` (benchmarks often use `1e-9`) |

**Kernel file:** all implementations expect `kernelInput.dat` in the **current working directory**.

---

## Outputs

Each run creates a timestamped output directory (naming varies by backend, e.g. `sim_serial_YYYYMMDD_HHMMSS/`).

Typical outputs include:

- `phi_XXXXXXXXXX.dat` — sampled 2D field
- `psi_XXXXXXXXXX.dat` — 1D profile
- `gw_XXXXXXXXXX.dat`, `gp_XXXXXXXXXX.dat` — static gradient profiles
- `intKernel.dat` — the (scaled) kernel used in the run
- (some implementations) `step_times.csv` — per-step timing diagnostics

**Important:** to keep output size manageable, some C++ implementations *subsample* outputs
(e.g. `sampleSkip = ceil(N/256)` before writing).

Benchmarks typically set `NO` very large to minimize disk I/O.

---

## Quick start

All backends assume `kernelInput.dat` is available via a **relative path**.
The simplest workflow is:

1) **Compile/build inside the subfolder** (if needed)  
2) **Run from the repo root**, calling the binary/script via its subfolder path

### C++ (serial)
```bash
g++ -O3 -std=c++17 cpp_serial/cpp_serial.cpp -o cpp_serial/sim_serial
./cpp_serial/sim_serial 256
```

### C++ (OpenMP parallel)
```bash
g++ -O3 -std=c++17 -fopenmp cpp_parallel/cpp_parallel.cpp -o cpp_parallel/sim_cpu
./cpp_parallel/sim_cpu 256
```

### C++/CUDA
```bash
# Compile the unity-build entry point. The included file selects the linear-gradient kernel:
#   cpp_cuda/src/cuda_kernel_linear.cu (default)
# Alternative available:
#   cpp_cuda/src/cuda_kernel_sigmoid.cu
nvcc -O3 -std=c++17 cpp_cuda/benchmark_main.cu -o cpp_cuda/sim_linear
./cpp_cuda/sim_linear 256
```

### MATLAB
From MATLAB, starting in the repo root:
```matlab
addpath('matlab_serial');
addpath('matlab_vectorized');
addpath('matlab_gpu_array');

run_conservative_transport(256);                 % matlab_serial/
run_conservative_transport_optimized(256);       % matlab_vectorized/
run_conservative_transport_gpu(256);             % matlab_gpu_array/ (needs NVIDIA GPU + PCT)
```

### Julia
From the repo root:
```bash
julia julia_cpu_serial/julia_cpu_serial.jl 256
julia julia_cpu_parallel/julia_cpu_parallel.jl 256
julia julia_cuda/julia_cuda.jl 256
```

Julia CUDA dependencies (install once):
```bash
julia -e 'using Pkg; Pkg.add(["CUDA", "Printf", "Dates", "DelimitedFiles"])'
```

---

## Benchmarking workflow

There are per-backend benchmark scripts (mostly Python or MATLAB) that run a sweep of grid sizes and parse
the printed line:

```
-> avg step time: <mean> ms (+/- <std> ms)
```

Typical sweep is `N = 128 … 8192` (powers of two). Many scripts scale `T` as `~ 1/N` (sometimes with an extra `/12`) to keep runs tractable.

### Plotting benchmark results

`benchmark_plotter.m` expects text tables like:

```
N    avg_ms    std_ms
```

Example files already present at repo root (e.g., `cpp_cpu.txt`, `cpp_cuda.txt`, …) match that convention.
If your benchmark script produces `.tsv`, you can rename or export to the expected `*.txt` files.

In MATLAB (repo root):
```matlab
benchmark_plotter
```

---

## Notes on reproducibility

- **Working directory matters:** most code loads `kernelInput.dat` using a relative path.
- **I/O dominates small runs:** use a very large `NO` to benchmark compute, not disk output.
- **CPU affinity / denormals:** the C++ serial version explicitly pins to core 0 and enables FTZ/DAZ (see `cpp_serial/cpp_serial.cpp`).

---

## Appendix — extracted subfolder READMEs (verbatim)

### `cpp_parallel/README.txt`
```text
# Linux / macOS (GCC/Clang)
g++ -O3 -std=c++17 -fopenmp simulation_cpu.cpp -o sim_cpu

# Run
./sim_cpu 256

Argument List (in order)Only the first argument (N) is mandatory. The rest are optional and will default to the values defined in the code if omitted.PositionVariableTypeDescriptionDefault (if omitted)1NintGrid size (Required)None2ISFdoubleInteraction Scale Factor1.0003PSIdoubleAvg RBC volume fraction0.024ITdoubleTime increment (dt)0.0055TdoubleTotal simulation time1200.06NOdoubleOutput interval (steps)10007DdoubleDiffusion coefficient0.0Usage Examples1. Minimal (Just Grid Size):Runs with $N=256$ and all other parameters at default.Bash./sim_cpu 256
2. Custom Physics (Grid + Physics):Runs with $N=512$, $ISF=1.0$, $PSI=0.05$, $dt=0.001$, TotalTime=500.0.Bash./sim_cpu 512 1.0 0.05 0.001 500.0
3. Full Configuration:Explicitly sets every parameter, including diffusion.Bash./sim_cpu 256 1.0 0.02 0.005 1200.0 500 1e-9
```

### `cpp_serial/README.txt`
```text
256 0.01 0.02 0.01 120.0 200 1e-9
g++ -O3 -std=c++17 cpp_serial.cpp -o sim_serial

./sim_serial 256 1.0 0.02 0.005 100

# Run on Core 0 with maximum priority (-20)
sudo nice -n -20 taskset -c 0 ./sim_serial 256

# Run on Core 0 only
taskset -c 0 ./sim_serial 256

g++ -O3 -std=c++17 -ffast-math simulation_serial.cpp -o sim_serial
```
