/*
    SI units for physical parameters
*/
// RBC parameters
double PSI = 0.02; // [v/v] RBC average volume fraction
double ISF = 1.000;
// time iteration parameters
double IT = 0.005;   // [s] time increment 
double T = 1200.0; // [s] total simulation time
double NO = 1000; // [steps] output interval 

// Diffusion parameter
double D_coeff = 0.0; // [m^2/s] Default to 0 (no diffusion)

// system parameters
int N;         // grid size (N x N) - set in readParameters
double subDiv; // set based on N
int M;         // set based on N
int wingL;     // set based on N
double SYS_L;  // set based on M
double zShift; // set based on SYS_L

// time iteration parameters
uint32_t NT; // set in readParameters

// spatial coordinate 
double IZ; //[m] space increment
double IR; //[g/L] density increment

// host flux prefactors
double h_beta = 7.4e23; // interaction integral
double h_alpha = 12e-5; // exp -4 for 20000 g, exp -5 for 2000 g
// stabilization flux (overhauled, use for experimentation)
double h_gamma = 3e-10; // degenerate diffusion restriction phi 0
double h_delta = 1e-15; // degenerate diffusion restriction psi 0
double h_kappa = 1e-15; // degenerate diffusion restriction psi 1
// interaction convolution kernel
double intKernel[kernelN]; // kernel array
// constant memory for convolution kernel coefficients
// The coefficients are copied from intKernel to this array once in runSim().
__constant__ double c_convKernel[kernelN];

// cuda device constants
__constant__ double c_IZ; // cuda space increment (Z)
__constant__ double c_IR; // cuda density increment (Rho)
__constant__ double c_IT; // cuda time increment
__constant__ double c_PSI; // cuda concentration
__constant__ double c_alpha; // sedimentation
__constant__ double c_beta; // interaction
__constant__ double c_gamma; // restriction phi 0
__constant__ double c_delta; // restriction psi 0
__constant__ double c_kappa; // restriction psi 1
__constant__ double c_D;     // cuda diffusion coefficient

// Dynamic grid constants for Device
__constant__ int c_N;
__constant__ int c_M;
__constant__ double c_subDiv;
__constant__ int c_wingL;
__constant__ double c_zShift;