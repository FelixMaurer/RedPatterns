/*
    SI units for physical parameters
*/
// RBC parameters
#define PSI 0.02 // [v/v] RBC average volume fraction
#define U 31.5 // [fJ] RBC effective interaction energy
// time iteration parameters
#define IT 0.005   // [s] time increment 
#define T 1200.0 // [s] total simulation time
#define NO 1000 // [steps] output interval 
// spatial dimension
#define XL (M*9.918212890625e-7) //[m] system length
// density dimension
#define RC 1100.0 // central density
#define RL 30.0 // density range (RC +- RL/2)
// interaction potential
#define kernelN 31 // kernel size
#define sigmaAtt 5.6E-6
#define sigmaRep 5.6E-6
#define nAtt 6
#define nRep 12
#define xi -0.0030877169233 // integral correction
#define epsilon 8e-7 // attractivity
#define epsilonAtt (0.25*(1+xi+epsilon)*nAtt/pow(sigmaAtt,3)/sqrt(PI)) // kernel prefactor
#define epsilonRep (0.25*(1-xi-epsilon)*nRep/pow(sigmaRep,3)/3.6256099082219083119306851558676720029951676828800654674333779995) // kernel prefactor
// flux terms
#define gamma 3e-10 // degenerate diffusion restriction
#define delta 1e-15 // degenerate diffusion restriction
// degenerate diffusion flux
#define jDegDiff(i) (gamma*pow(1.0-phi[i]/PSI*0.05,500))
#define jDegDiff1(i) (-delta*pow(psi[i],500))
// initial RBC density function
#define Rsigma 4.0f // [g/l] gaussian width
#define Rmu 1100.0f // [g/l] central RBC density   
// interpolation
#define subDiv 256.0 // subdivision
#define M int(N*subDiv + 1) // size of interpolated grid
// Percoll density gradient
#define gradL 0.06 // [m] tube length
#define b1 1100.0 // [g/l] central PC density
#define b2 3.1773e-4
#define b3 (gradL/2)
#define b4 0.0338
#define b5 1.1012e-3
#define b6 0.6
#define b7 1.5205
#define wingL 15 // [grid] length of gradient wings
#define xShift ((XL-gradL)/2)
// misc
#define PI 3.141592653589793115997963468544185161590576171875