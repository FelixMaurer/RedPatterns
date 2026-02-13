/*
    SI units for physical parameters
*/
// spatial dimension
#define sysL (M*1.041412353515625e-6) //[m] system length
// density dimension
#define RC 1100.0 // central density
#define RL 30.0 // density range (RC +- RL/2)
// interaction potential
#define kernelN 31 // kernel size
// degenerate diffusion flux
#define mDeg 500
#define jDegDiffPhi0(i) (pow(1.0-phi[i],mDeg))
/*
 * The degenerate diffusion flux is obsolete and overhauled by intrinsically 
 * stable flux computations.   
 * NOTE: The degenerate diffusion terms for psi are now computed using
 * precomputed power factors (psiPow0 and psiPow1) inside the iteration
 * kernel. The original macros jDegDiffPsi0 and jDegDiffPsi1 have been
 * removed to avoid inadvertently using expensive pow() operations on the
 * device within the iteration loop. See CuKernelIter implementation for
 * details.
 */
/* #define jDegDiffPsi0(gi,i) (pow(1.0-psi[i],mDeg)*abs(phi[gi])) */
/* #define jDegDiffPsi1(gi,i) ((-pow(psi[i],mDeg))*abs(phi[gi])) */
// initial RBC density function
#define Rsigma 4.0f // [g/l] gaussian width
#define Rmu 1100.0f // [g/l] central RBC density   
// interpolation
#define subDiv 256.0 // subdivision
#define M int(N*subDiv + 1) // size of interpolated grid
// Percoll density gradient
#define gradL 0.06 // [m] tube length
#define wingL 30 // [grid] length of gradient wings
#define zShift ((sysL-gradL)/2) // gradient spatial center
#define P0 1100.0 // [g/l] central PC density
// misc
#define PI 3.141592653589793115997963468544185161590576171875