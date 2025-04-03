/*
    SI units for physical parameters
*/
// RBC parameters
double PSI = 0.02; // [v/v] RBC average volume fraction
double U = 100e-18; // [J] RBC effective interaction energy
// time iteration parameters
double IT = 0.005;   // [s] time increment 
double T = 1200.0; // [s] total simulation time
double NO = 1000; // [steps] output interval 
// system parameters
const int N = 256; // grid size (N x N)
// time iteration parameters
uint32_t NT = ceil(T/IT); // number of time steps
// spatial coordinate 
const double IZ = sysL/(N-1); //[m] space increment
// host flux prefactors
double h_beta = 7.4e23; // interaction integral
double h_alpha = 2e-4; // exp -4 for 20000 g, exp -5 for 2000 g
double h_gamma = 3e-10; // degenerate diffusion restriction phi 0
double h_delta = 1e-15; // degenerate diffusion restriction psi 0
double h_kappa = 1e-15; // degenerate diffusion restriction psi 1
// interaction convolution kernel
double intKernel[kernelN]; // kernel array
// cuda device constants
__constant__ double c_IZ; // cuda space increment
__constant__ double c_IT; // cuda time increment
__constant__ double c_PSI; // cuda concentration
__constant__ double c_alpha; // sedimentation
__constant__ double c_beta; // interaction
__constant__ double c_gamma; // restriction phi 0
__constant__ double c_delta; // restriction psi 0
__constant__ double c_kappa; // restriction psi 1