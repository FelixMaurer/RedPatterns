RedPatterns (CUDA) — Conservative FVM Solver + External Kernel Pipeline

2025 Felix Maurer, Experimentalphysik AG Wagner, Universität des Saarlandes



Manuscript: https://www.pnas.org/doi/10.1073/pnas.2515704122





1\) What this code does

----------------------

RedPatterns simulates pattern formation / banding dynamics in RBC suspensions on a 2D grid

(N x N) using CUDA acceleration. The current version uses a conservative finite-volume

transport update (flux form) with an upwind scheme for stability.



Key upgrades vs older versions:

\- Conservative finite-volume method (FVM): compute interface fluxes, then update cell values.

\- Sealed boundaries (zero flux at both ends) for strict mass conservation.

\- Upwind differencing for advection to eliminate central-difference ringing.

\- Kernel generation moved to Python (createKernel.py) and loaded from disk at runtime.

\- Each run writes to its own timestamped output folder (no overwrites).





2\) Repository layout (important files)

--------------------------------------

main.cu

&nbsp; Entry point. Detects GPU, reads CLI parameters, runs simulation.



src/

&nbsp; CUDA kernels and core simulation routines.



createKernel.py

&nbsp; Generates the interaction kernel and writes it to kernelInput.dat.



kernelInput.dat

&nbsp; Precomputed kernel (tab-separated, single row). The executable loads this at startup.



colorMapModel.mat

&nbsp; Optional: mapping from psi (volume fraction) to RGB, based on photographic calibration.



condor\_docker\_interactive.sub / condor\_docker\_queue.sub

&nbsp; HTCondor helpers (NOTE: queue submit file may require argument updates; see Section 7).



runSlurmJob.sh

&nbsp; Slurm helper (cluster-specific; adapt as needed).



plot\_psi\_t.py + gnu\_plot\_script

&nbsp; Example post-processing using awk + gnuplot (run inside an output directory).





3\) Requirements

---------------

\- NVIDIA GPU with CUDA support

\- nvcc / CUDA toolkit installed

\- Python 3 + numpy (for createKernel.py)

\- (Optional) gnuplot (for plot\_psi\_t.py workflow)





4\) Quick start (local)

----------------------



Step A — Generate kernelInput.dat (recommended)

Edit createKernel.py to set the physical kernel parameters (notably U, IZ, kernelN),

then run:



&nbsp; python3 createKernel.py



This writes:

&nbsp; kernelInput.dat



Notes on parameters:

\- U is defined inside createKernel.py (absolute interaction strength used for kernel construction).

\- The simulation executable does NOT take U directly anymore (see ISF below).

\- IZ in createKernel.py is intended to match the \*fine grid spacing\* used in the convolution.

&nbsp; In the CUDA code the convolution integrates with (c\_IZ / subDiv), where:

&nbsp;   c\_IZ = sysL/(N-1)   and fine\_spacing ≈ c\_IZ/subDiv



If you change N, subDiv, or the physical length scaling, re-check IZ in createKernel.py.



Step B — Compile

Select the correct SM architecture for your GPU.



Example (Tesla P100, sm\_60):

&nbsp; nvcc -Xptxas -O3 -gencode arch=compute\_60,code=\[sm\_60,compute\_60] main.cu -o red\_patterns



Example (A100, sm\_80):

&nbsp; nvcc -Xptxas -O3 -gencode arch=compute\_80,code=\[sm\_80,compute\_80] main.cu -o red\_patterns



Step C — Run

The executable expects kernelInput.dat in the working directory.



CLI:

&nbsp; ./red\_patterns ISF PSI IT T NO



Where:

\- ISF : Interaction Scale Factor (dimensionless). Scales the loaded kernel linearly.

\- PSI : Mean RBC volume fraction \[v/v] (e.g., 0.02)

\- IT  : Time step \[s] (e.g., 5e-4)

\- T   : Total simulation time \[s] (e.g., 1200)

\- NO  : Output interval in iteration steps (integer)



Example:

&nbsp; ./red\_patterns 1.0 0.02 2e-3 1200.0 3000



Defaults:

If you omit arguments, defaults are taken from src/constants.cu.





5\) Output format and directories

--------------------------------

Each run creates a new timestamped output directory:



&nbsp; sim\_YYYYMMDD\_HHMMSS/



Inside you will find:

\- phi\_##########.dat   (N x N array, tab-separated)

\- psi\_##########.dat   (N vector, tab-separated)

\- gw\_##########.dat    (N vector, tab-separated; “gradient wing” / external gradient term)

\- gp\_##########.dat    (N vector, tab-separated; percoll / PC density gradient term)

\- intKernel.dat        (kernel actually used, after scaling by ISF; saved with max double precision)



Output cadence:

\- The simulation writes at iteration i=1 and then every NO steps according to the internal check.

&nbsp; (Practically: 1, 1+NO, 1+2\*NO, ...)



Precision:

\- intKernel.dat is saved with full double round-trip precision

&nbsp; (scientific notation, max\_digits10).





6\) Parameter meaning (at a glance)

----------------------------------

ISF (Interaction Scale Factor)

\- This is the primary sweep knob at runtime.

\- The kernel produced by createKernel.py is treated as a baseline; ISF scales it.



PSI

\- Mean volume fraction used by the model.



IT, T, NT

\- IT is the time step.

\- T is total time.

\- NT = ceil(T/IT) is computed internally.



NO

\- Output interval (in iteration steps).



Grid size and geometry

\- N is compile-time (const int N = 256 in src/constants.cu).

\- subDiv and M are set in src/definitions.h.

&nbsp; If you change N/subDiv you must recompile, and you should also verify kernel generation settings.





7\) Cluster usage (HTCondor / Slurm)

-----------------------------------



HTCondor interactive (quick debug)

1\) Start an interactive container session:

&nbsp;  condor\_submit -i condor\_docker\_interactive.sub



2\) Inside the session:

&nbsp;  - ensure kernelInput.dat is present

&nbsp;  - compile (nvcc ...)

&nbsp;  - run (./red\_patterns ISF PSI IT T NO)



HTCondor queued runs (IMPORTANT NOTE)

\- condor\_docker\_queue.sub already transfers kernelInput.dat via:

&nbsp;   transfer\_input\_files = kernelInput.dat



\- HOWEVER: the example "arguments =" line in the submit file may still follow the legacy 8-argument

&nbsp; format from older versions. The new executable expects ONLY:

&nbsp;   ISF PSI IT T NO



So update the submit file accordingly, e.g. if you sweep IT:

&nbsp; arguments = 1.0 0.02 $(step\_size) 1200.0 3000



Slurm

Typical interactive run pattern:

&nbsp; srun -J RedPatterns --partition=<...> --time=00:10:00 --ntasks=1 --cpus-per-task=1 \\

&nbsp;      --gpus-per-task=1 --nodes=1 --mem=6GB --pty /bin/bash



Then compile + run as in Section 4.

Make sure kernelInput.dat is present in the working directory.





8\) Post-processing examples

---------------------------

The provided plot script expects psi\*.dat in the current working directory.

Since outputs are now in sim\_\*/ directories, do:



&nbsp; cd sim\_YYYYMMDD\_HHMMSS/

&nbsp; python3 ../plot\_psi\_t.py



This creates:

&nbsp; psi\_t.dat

&nbsp; psi\_t.png



Color mapping (psi -> RGB)

\- See colorMapModel.mat for the calibration fit functions and coefficients.





9\) Notes on the numerical method (for users modifying the solver)

-----------------------------------------------------------------

\- Transport update is conservative (finite volume): net flux divergence updates phi.

\- Upwind sampling is used for advective fluxes based on face velocity sign.

\- Boundary flux is enforced to 0 at both ends (sealed box) to prevent mass drift.

\- Legacy “degenerate diffusion” stabilization infrastructure is not part of the runtime update path.





10\) Citation

------------

If you use this code, please cite the associated manuscript:

https://www.pnas.org/doi/10.1073/pnas.2515704122


