2025 Felix Maurer
Experimentalphysik AG Wagner
Universität des Saarlandes

RedPatterns script optimized for CUDA computing. 
Manuscript available at https://doi.org/10.48550/arXiv.2407.07676.

compile with
nvcc -Xptxas -O3 -gencode arch=compute_60,code=[sm_60,compute_60] main.cu -o red_patterns
(select cuda compatibility for your device, 60 in this example)

Condor fast run:
(1) interactive session in condor with 
condor_submit -i condor_docker_interactive.sub
(2) compilation
(3) ./red_patterns 111.15e-18 0.02 5.00000000e-04 1200.0 3000 1.80000000e-10 1e-11 0

tested on Tesla P100-PCIE-16GB (5300 s runtime) and 

Slurm fast run: 
(1) launch interactive session 
srun -J RedPatterns --account=nhr-rbc-pattern --partition=a100ai --time=00:10:00 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --nodes=1 --mem=6GB --pty /bin/bash
(2) compile
(3) ./red_patterns 111.15e-18 0.02 5.00000000e-04 1200.0 3000 1.80000000e-10 1e-11 0