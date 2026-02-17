/*
    SI units for physical parameters
*/
// spatial dimension
// sysL is now calculated in constants/functions based on M
extern double SYS_L; //[m] system length

// density dimension
#define RC 1100.0 // central density
#define RL 30.0 // density range (RC +- RL/2)
// interaction potential
#define kernelN 31 // kernel size
// degenerate diffusion flux
#define mDeg 500
#define jDegDiffPhi0(i) (pow(1.0-phi[i],mDeg))

// initial RBC density function
#define Rsigma 2.0f // [g/l] gaussian width
#define Rmu 1100.0f // [g/l] central RBC density   

// interpolation
// subDiv and M are now dynamic externs
extern double subDiv; 
extern int M; // size of interpolated grid

// Percoll density gradient
#define gradL 0.06 // [m] tube length
// wingL and zShift are now dynamic externs
extern int wingL;
extern double zShift;

#define P0 1100.0 // [g/l] central PC density

// Diffusion
extern double D_coeff; // Stabilizing diffusion coefficient

// misc
#define PI 3.141592653589793115997963468544185161590576171875