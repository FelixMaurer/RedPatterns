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

tested on Tesla P100-PCIE-16GB (5300 s runtime)

Slurm fast run: 
(1) launch interactive session 
srun -J RedPatterns --account=nhr-rbc-pattern --partition=a100ai --time=00:10:00 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --nodes=1 --mem=6GB --pty /bin/bash
(2) compile
(3) ./red_patterns 111.15e-18 0.02 5.00000000e-04 1200.0 3000 1.80000000e-10 1e-11 0


ColorMap: convert output volume fraction psi to RGB triplet
The color map is based on photographic measurements of RBC suspensions. 
It can be found in the colorMapModel.mat file. 
This file contains cfit functions for use in matlab but there is also an explicit expression of a model function. 

@(b,x)real(b(1)+b(2)*(x-b(3))/b(4)./(1+((x-b(3))/b(4)).^b(5)).^(1/b(5)))

The values for vector b for R, G, and B channel are: 
      R           G            B
b(1)  145.7586    134.7227     130.9048
b(2)  -130.4334   -124.1164    -101.2598
b(3)  0.8120      -0.0642      -0.1119
b(4)  0.7014      0.3524       0.4840
b(5)  2.4949      2.2571       5.9188

