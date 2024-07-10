/*
    SI units for physical parameters
*/
// system parameters
const int N=256; // grid size (N x N)
// time iteration parameters
const uint32_t NT = ceil(T/IT); // number of time steps
// spatial coordinate 
const double IX = XL/(N-1); //[m] space increment
// flux prefactors
const double h_beta = -U*0.1075; // interaction integral
const double h_alpha = 2e-5f; // exp -4 for 20000 g, exp -5 for 2000 g
// interaction convolution kernel
double intKernel[kernelN]; // kernel array
// cuda device constants
__constant__ double c_IX; // cuda space increment
__constant__ double c_alpha; 
__constant__ double c_beta; 