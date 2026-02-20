# RedPatterns (CUDA) — Conservative FVM Solver + External Kernel Pipeline

2025 Felix Maurer, Experimentalphysik AG Wagner, Universität des Saarlandes

Manuscript: https://www.pnas.org/doi/10.1073/pnas.2515704122

---

## 1) What this code does

RedPatterns simulates pattern formation / banding dynamics in RBC suspensions on a 2D grid (N x N) using CUDA acceleration. The current version uses a conservative finite-volume transport update (flux form) with an upwind scheme for stability.

**Key upgrades vs older versions:**
- Conservative finite-volume method (FVM): compute interface fluxes, then update cell values.
- Sealed boundaries (zero flux at both ends) for strict mass conservation.
- Upwind differencing for advection to eliminate central-difference ringing.
- Kernel generation moved to Python (`createKernel.py`) and loaded from disk at runtime.
- Each run writes to its own timestamped output folder (no overwrites).

---

## 2) Repository layout (important files)

- `main.cu`  
  Entry point. Detects GPU, reads CLI parameters, runs simulation.

- `src/`  
  CUDA kernels and core simulation routines.

- `createKernel.py`  
  Generates the interaction kernel and writes it to `kernelInput.dat`.

- `kernelInput.dat`  
  Precomputed kernel (tab-separated, single row). The executable loads this at startup.

- `colorMapModel.mat`  
  Optional: mapping from psi (volume fraction) to RGB, based on photographic calibration.

- `condor_docker_interactive.sub` / `condor_docker_queue.sub`  
  HTCondor helpers (NOTE: queue submit file may require argument updates; see Section 7).

- `runSlurmJob.sh`  
  Slurm helper (cluster-specific; adapt as needed).

- `plot_psi_t.py` + `gnu_plot_script`  
  Example post-processing using awk + gnuplot (run inside an output directory).

---

## 3) Requirements

- NVIDIA GPU with CUDA support
- `nvcc` / CUDA toolkit installed
- Python 3 + numpy (for `createKernel.py`)
- (Optional) gnuplot (for `plot_psi_t.py` workflow)

---

## 4) Quick start (local)

### Step A — Generate `kernelInput.dat` (recommended)

Edit `createKernel.py` to set the physical kernel parameters (notably `U`, `IZ`, `kernelN`), then run:

```bash
python3 createKernel.py
```

This writes:

* `kernelInput.dat`

**Notes on parameters:**

* `U` is defined inside `createKernel.py` (absolute interaction strength used for kernel construction).
* The simulation executable does **NOT** take `U` directly anymore (see `ISF` below).
* `IZ` in `createKernel.py` is intended to match the *fine grid spacing* used in the convolution.
  In the CUDA code the convolution integrates with `(c_IZ / subDiv)`, where:

  * `c_IZ = sysL/(N-1)` and fine_spacing ≈ `c_IZ/subDiv`

If you change `N`, `subDiv`, or the physical length scaling, re-check `IZ` in `createKernel.py`.

### Step B — Compile

Select the correct SM architecture for your GPU.

Example (Tesla P100, `sm_60`):

```bash
nvcc -Xptxas -O3 -gencode arch=compute_60,code=[sm_60,compute_60] main.cu -o red_patterns
```

Example (A100, `sm_80`):

```bash
nvcc -Xptxas -O3 -gencode arch=compute_80,code=[sm_80,compute_80] main.cu -o red_patterns
```

### Step C — Run

The executable expects `kernelInput.dat` in the working directory.

CLI:

```bash
./red_patterns ISF PSI IT T NO
```

Where:

* `ISF` : Interaction Scale Factor (dimensionless). Scales the loaded kernel linearly.
* `PSI` : Mean RBC volume fraction [v/v] (e.g., 0.02)
* `IT`  : Time step [s] (e.g., 5e-4)
* `T`   : Total simulation time [s] (e.g., 1200)
* `NO`  : Output interval in iteration steps (integer)

Example:

```bash
./red_patterns 1.0 0.02 2e-3 1200.0 3000
```

**Defaults:**
If you omit arguments, defaults are taken from `src/constants.cu`.

---

## 5) Output format and directories

Each run creates a new timestamped output directory:

* `sim_YYYYMMDD_HHMMSS/`

Inside you will find:

* `phi_##########.dat`   (N x N array, tab-separated)
* `psi_##########.dat`   (N vector, tab-separated)
* `gw_##########.dat`    (N vector, tab-separated; “gradient wing” / external gradient term)
* `gp_##########.dat`    (N vector, tab-separated; percoll / PC density gradient term)
* `intKernel.dat`        (kernel actually used, after scaling by ISF; saved with max double precision)

**Output cadence:**

* The simulation writes at iteration `i=1` and then every `NO` steps according to the internal check.
  (Practically: `1, 1+NO, 1+2*NO, ...`)

**Precision:**

* `intKernel.dat` is saved with full double round-trip precision (scientific notation, `max_digits10`).

---

## 6) Parameter meaning (at a glance)

### ISF (Interaction Scale Factor)

* This is the primary sweep knob at runtime.
* The kernel produced by `createKernel.py` is treated as a baseline; `ISF` scales it.

### PSI

* Mean volume fraction used by the model.

### IT, T, NT

* `IT` is the time step.
* `T` is total time.
* `NT = ceil(T/IT)` is computed internally.

### NO

* Output interval (in iteration steps).

### Grid size and geometry

* `N` is compile-time (`const int N = 256` in `src/constants.cu`).
* `subDiv` and `M` are set in `src/definitions.h`.
  If you change `N/subDiv` you must recompile, and you should also verify kernel generation settings.

---

## 7) Cluster usage (HTCondor / Slurm)

### HTCondor interactive (quick debug)

1. Start an interactive container session:

```bash
condor_submit -i condor_docker_interactive.sub
```

2. Inside the session:

* ensure `kernelInput.dat` is present
* compile (`nvcc ...`)
* run (`./red_patterns ISF PSI IT T NO`)

### HTCondor queued runs (IMPORTANT NOTE)

* `condor_docker_queue.sub` already transfers `kernelInput.dat` via:

  * `transfer_input_files = kernelInput.dat`

* HOWEVER: the example `arguments =` line in the submit file may still follow the legacy 8-argument format from older versions. The new executable expects ONLY:

  * `ISF PSI IT T NO`

So update the submit file accordingly, e.g. if you sweep `IT`:

* `arguments = 1.0 0.02 $(step_size) 1200.0 3000`

### Slurm

Typical interactive run pattern:

```bash
srun -J RedPatterns --partition=<...> --time=00:10:00 --ntasks=1 --cpus-per-task=1 \
     --gpus-per-task=1 --nodes=1 --mem=6GB --pty /bin/bash
```

Then compile + run as in Section 4.
Make sure `kernelInput.dat` is present in the working directory.

---

## 8) Post-processing and Plotting

To visualize the simulation results, three dedicated plotting scripts are provided (Python, Gnuplot, and MATLAB). These scripts read the 1D `psi*.dat` spatial arrays across all time steps, stack them into a 2D spacetime matrix (Time vs. Space), and render a high-quality SVG heatmap. 

Crucially, all three scripts contain the embedded mathematical functions and coefficients for the custom photographic color calibration, removing the need to load the external `colorMapModel.mat` file.

### Option A: Python (Recommended for Clusters)
The Python script is optimized for headless Linux environments. It forces Matplotlib to use the non-interactive `Agg` backend, preventing display errors on servers without GUIs.

* **Requirements**: Python 3, `numpy`, and `matplotlib`. 
* **Usage**: Pass the target simulation directory as a command-line argument.
  ```bash
  python plotter_python.py sim_YYYYMMDD_HHMMSS
  ```
* **Output**: Generates `sim_YYYYMMDD_HHMMSS.svg` in your current working directory.

### Option B: Gnuplot (Fastest & Lightest)
The Gnuplot script provides a highly efficient, native command-line plotting method. It uses internal system calls (`ls -v` and `cat`) to dynamically stitch the 1D `.dat` files into a 2D surface at runtime.

* **Requirements**: `gnuplot` and standard Linux utilities (`tr`, `cat`, `ls`).
* **Usage**: Execute using the `-c` flag to pass the directory as an argument.
  ```bash
  gnuplot -c plotter_gnuplot.gp sim_YYYYMMDD_HHMMSS
  ```
* **Output**: Generates `sim_YYYYMMDD_HHMMSS.svg` in your current working directory.

### Option C: MATLAB
If you prefer a local GUI workflow, the MATLAB script provides identical outputs using native MATLAB rendering.

* **Usage**: Open `plotter_matlab.m` in the MATLAB editor. Manually update the `simDirName` variable at the top of the script to match your target directory.
  ```matlab
  % Adjust this variable to point to your target simulation directory
  simDirName = 'sim_YYYYMMDD_HHMMSS'; 
  ```
* **Output**: Run the script to generate and save the SVG file natively.

---

## 9) Notes on the numerical method (for users modifying the solver)

* Transport update is conservative (finite volume): net flux divergence updates phi.
* Upwind sampling is used for advective fluxes based on face velocity sign.
* Boundary flux is enforced to 0 at both ends (sealed box) to prevent mass drift.
* Legacy “degenerate diffusion” stabilization infrastructure is not part of the runtime update path.

---

## 10) Citation

If you use this code, please cite the associated manuscript:

```
[https://www.pnas.org/doi/10.1073/pnas.2515704122](https://www.pnas.org/doi/10.1073/pnas.2515704122)
```
