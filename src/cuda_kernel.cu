/*
    This contains main CUDA kernels.
    -> integration (Inte) 
    -> interpolation (CmpA) 
    -> interpolation (CmpL)
    -> convolution (Conv) 
    -> downsampling (DSmp) 
    -> iteration (Iter)
*/
// Include math header for intrinsic functions such as fabs() and floor()
#include <math.h>
/* phi density integration kernel */
__global__ void CuKernelInte(double* phi, double* psi)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // Only compute for valid indices; threads beyond N perform no work.
    if (i < N) {
        double sum = 0.0;
        // discrete sum integration over the second dimension
        for(int k=0; k<N; k++) {
            sum += phi[k*N + i];
        }
        psi[i] = sum;
    }
}
/* kernels for cubic interpolation */
__global__ void CuKernelCmpA(double* y, double* alp)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // compute finite difference coefficients only for valid indices
    if(i >= 1 && i < N-1) {
        alp[i] = 3.0 * (y[i+1] - y[i]) - 3.0 * (y[i] - y[i-1]);
    }
}
__global__ void CuKernelCmpL(double* y, double* alp, double* psiIntp)
{
    // The original cubic spline interpolation is replaced by an optimized two-stage
    // spline implementation in CuKernelSplineCoeffs and CuKernelSplineEval. This
    // kernel is retained for backward compatibility but performs no real work.
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k < M) {
        psiIntp[k] = 0.0;
    }
}
/* convolution kernel */
__global__ void CuKernelConv(double* psi, double* I, double* convKernel)
{
    // Optimized 1D convolution kernel using constant memory for the kernel
    // coefficients and shared memory tiling for input data. The convKernel
    // parameter is unused but kept for signature compatibility.
    extern __shared__ double s_psi[];
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int halo = (kernelN - 1) / 2;
    int blockStart = blockIdx.x * blockDim.x;
    // load central data
    if (gid < M) {
        s_psi[tid + halo] = psi[gid];
    } else {
        s_psi[tid + halo] = 0.0;
    }
    // load left halo
    if (tid < halo) {
        int leftIdx = blockStart + tid - halo;
        s_psi[tid] = (leftIdx >= 0 ? psi[leftIdx] : 0.0);
    }
    // load right halo
    if (tid >= blockDim.x - halo) {
        int offset = tid - (blockDim.x - halo);
        int rightIdx = blockStart + tid + halo;
        int sIdx = halo + blockDim.x + offset;
        s_psi[sIdx] = (rightIdx < M ? psi[rightIdx] : 0.0);
    }
    __syncthreads();
    // perform convolution if in bounds
    if (gid < M) {
        double acc = 0.0;
        for (int k = 0; k < kernelN; ++k) {
            acc += s_psi[tid + k] * c_convKernel[k];
        }
        // apply scale factor outside the loop for efficiency
        I[gid] = acc * (c_IZ / subDiv);
    }
}
/* downsampling kernel */
__global__ void CuKernelDSmp(double* IIntp, double* I)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = i*subDiv;
    if (i < N) {
        I[i] = IIntp[j];
    }
}
/* main time iteration */
__global__ void CuKernelIter(double *phi, double *J, double *dJ, double* percoll, double *R, double* I, double* psi, double* psiPow0, double* psiPow1, double t, double* gradWing)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int gi = i + j*N; // global index

    // compute physical flux
    double rpTerm;
    if((i>wingL) & (i < N-1-wingL)) {
        rpTerm = R[j] + percoll[i] - P0;
    }
    if((i<=wingL) || (i >= N-1-wingL)) {
        rpTerm = R[j] + gradWing[i] - P0;
    }
    J[gi] = (c_alpha * rpTerm + c_beta * I[i]) * phi[gi];
    // compute flux derivative
    if((i>=4) && (i<=N-1-4)){
        // physical flux first derivative
        dJ[gi] = (0.5 / c_IZ) * ( J[gi+1] - J[gi-1] );
        // degenerate diffusion second derivative for phi (no precomputation needed)
        double ddPhi0_i   = jDegDiffPhi0(gi);
        double ddPhi0_p1  = jDegDiffPhi0(gi+1);
        double ddPhi0_m1  = jDegDiffPhi0(gi-1);
        dJ[gi] -= ( -2.0 * ddPhi0_i + (ddPhi0_p1 + ddPhi0_m1) ) / (c_IZ*c_IZ) * c_gamma;
        // degenerate diffusion second derivative for psi using precomputed power arrays
        double deg0     = psiPow0[i]   * fabs(phi[gi]);
        double deg0_p1  = psiPow0[i+1] * fabs(phi[gi+1]);
        double deg0_m1  = psiPow0[i-1] * fabs(phi[gi-1]);
        dJ[gi] -= ( -2.0 * deg0 + (deg0_p1 + deg0_m1) ) / (c_IZ*c_IZ) * c_delta;
        double deg1     = -psiPow1[i]   * fabs(phi[gi]);
        double deg1_p1  = -psiPow1[i+1] * fabs(phi[gi+1]);
        double deg1_m1  = -psiPow1[i-1] * fabs(phi[gi-1]);
        dJ[gi] += ( -2.0 * deg1 + (deg1_p1 + deg1_m1) ) / (c_IZ*c_IZ) * c_kappa;
    }
    // compute euler step
    phi[gi] = phi[gi] + c_IT * dJ[gi]; 
}

/*
 * Compute spline coefficients from the input array `y` and its first derivative estimates `alp`.
 * The coefficients `b`, `c` and `d` are written to global arrays of length N on the device.
 * This kernel uses a single thread to perform the Thomas algorithm and back-substitution exactly
 * as in the original implementation, preserving arithmetic order to ensure identical results.
 */
__global__ void CuKernelSplineCoeffs(double* y, double* alp, double* b, double* c, double* d)
{
    // Use a single thread for the entire computation.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double mu[N];
        double ze[N];
        mu[0] = 0.0;
        ze[0] = 0.0;
        for (int i = 1; i < N - 1; ++i) {
            mu[i] = 1.0 / (4.0 - mu[i-1]);
            ze[i] = (alp[i] - ze[i-1]) / (4.0 - mu[i-1]);
        }
        ze[N-1] = 0.0;
        mu[N-1] = 0.0;
        double localC[N];
        double localB[N];
        double localD[N];
        localC[N-1] = 0.0;
        for (int j = N - 2; j >= 0; --j) {
            localC[j] = ze[j] - mu[j] * localC[j + 1];
            // (y[j+1] - y[j]) / 1.0 - 1.0 * (c[j+1] + 2.0 * c[j]) / 3.0
            localB[j] = (y[j+1] - y[j]) - (localC[j+1] + 2.0 * localC[j]) / 3.0;
            // (c[j+1] - c[j]) / (3.0 * 1.0)
            localD[j] = (localC[j+1] - localC[j]) / 3.0;
        }
        for (int idx = 0; idx < N; ++idx) {
            b[idx] = localB[idx];
            c[idx] = localC[idx];
            d[idx] = localD[idx];
        }
    }
}

/*
 * Evaluate the cubic spline defined by `y`, `b`, `c` and `d` on a uniformly subdivided grid of size M.
 * For each k in [0, M-1], the interpolated value psiIntp[k] is computed exactly as in the original code.
 */
__global__ void CuKernelSplineEval(double* y, double* b, double* c, double* d, double* psiIntp)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < M) {
        double x  = double(k) / subDiv;
        int j     = floor(x);
        double dx = x - double(j);
        psiIntp[k] = y[j] + (b[j] + (c[j] + d[j] * dx) * dx) * dx;
    }
}

/*
 * Precompute degenerate diffusion power factors. For each i in [0, N-1],
 * psiPow0[i] = pow(1.0 - psi[i], mDeg) and psiPow1[i] = pow(psi[i], mDeg).
 * This kernel runs with a 1D grid over N elements.
 */
__global__ void CuKernelDegDiffPow(double* psi, double* psiPow0, double* psiPow1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double val = psi[i];
        psiPow0[i] = pow(1.0 - val, mDeg);
        psiPow1[i] = pow(val, mDeg);
    }
}