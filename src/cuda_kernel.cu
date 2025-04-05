/*
    This contains main CUDA kernels.
    -> integration (Inte) 
    -> interpolation (CmpA) 
    -> interpolation (CmpL)
    -> convolution (Conv) 
    -> downsampling (DSmp) 
    -> iteration (Iter)
*/
/* phi density integration kernel */
__global__ void CuKernelInte(double* phi, double* psi)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    double sum = 0.0;
    __syncthreads();
    // discrete sum integration
    for(int k=0; k<N; k++)
        sum += phi[(k)*N+i];
    __syncthreads();
    psi[i] = sum;
}
/* kernels for cubic interpolation */
__global__ void CuKernelCmpA(double* y, double* alp)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    __syncthreads();
    if(i >= 1 &  i < N-1) 
        alp[i] = 3.0 * (y[i+1] - y[i]) / 1.0 - 3.0 * (y[i] - y[i-1]) / 1.0;
}
__global__ void CuKernelCmpL(double* y, double* alp, double* psiIntp)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    double mu[N], ze[N];
    mu[0] = 0;
    ze[0] = 0;
    for (int i = 1; i < N-1; ++i) {
        mu[i] = 1/(4.0-mu[i-1]);
        ze[i] = (alp[i]-ze[i-1])/(4.0-mu[i-1]);
    }
    ze[N-1] = 0;
    mu[N-1] = 0;
    double d[N], b[N], c[N];
    c[N-1] = 0;
    for (int j = N-2; j>=0; --j) {
        c[j] = ze[j] - mu[j] * c[j + 1];
        b[j] = (y[j+1] - y[j]) / 1.0 - 1.0 * (c[j+1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j+1] - c[j]) / (3.0 * 1.0);
    }
    double x; 
    double dx;
    int j;
    __syncthreads();
    x = double(k)/subDiv;
    j = floor(x);
    dx = x-j;
    psiIntp[k] = y[j] + (b[j] + (c[j] + d[j] * dx) * dx) * dx;
    __syncthreads();
}
/* convolution kernel */
__global__ void CuKernelConv(double* psi, double* I, double* convKernel)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // compute convolution integral
    double sum = 0.0;
    int d = (kernelN-1)/2;
    __syncthreads();
    for(int k=0; k<kernelN; k++)
        if((i+(k-d) >= 0)&(i+(k-d)<M))
        {   
            sum += psi[i+(k-d)]*convKernel[k]*(c_IZ/subDiv);
        }
    I[i] = sum;
    __syncthreads();
}
/* downsampling kernel */
__global__ void CuKernelDSmp(double* IIntp, double* I)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = i*subDiv;
    I[i] = 0;
    __syncthreads();
    I[i] = IIntp[j];
    __syncthreads();
}
/* main time iteration */
__global__ void CuKernelIter(double *phi, double *J, double *dJ, double* percoll,double *R, double* I, double* psi, double* convKernel,double t,double* gradWing)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int gi = i + j*N; // global index

    // compute physical flux
    double rpTerm;
    if((i>wingL) & (i < N-1-wingL)) 
        rpTerm = R[j]+percoll[i]-P0;
    __syncthreads();
    if((i<=wingL) | (i >= N-1-wingL))
        rpTerm = R[j]+gradWing[i]-P0;
    __syncthreads();
    J[gi] = (c_alpha*rpTerm + c_beta*I[i])*phi[gi];
    __syncthreads();
    // compute flux derivative
    if((i>=4) & (i<=N-1-4)){
        // physical flux first derivative
        dJ[gi] = (
            +  0.5/c_IZ * ( J[gi+1] - J[gi-1] )
            );
        // degenerate diffusion second derivative phi 0
        dJ[gi] -= (
            - 2.0 * ( jDegDiffPhi0(gi) )
            + 1.0 * ( jDegDiffPhi0(gi+1)+jDegDiffPhi0(gi-1) )
            )/(c_IZ*c_IZ)*c_gamma;
        // degenerate diffusion second derivative psi 0
        dJ[gi] -= (
            - 2.0 * ( jDegDiffPsi0(gi,i) )
            + 1.0 * ( jDegDiffPsi0(gi,i+1)+jDegDiffPsi0(gi,i-1) )
            )/(c_IZ*c_IZ)*c_delta;
        // degenerate diffusion second derivative psi 1
        dJ[gi] += (
            - 2.0 * ( jDegDiffPsi1(gi,i) )
            + 1.0 * ( jDegDiffPsi1(gi,i+1)+jDegDiffPsi1(gi,i-1) )
            )/(c_IZ*c_IZ)*c_kappa;
    }
    __syncthreads();
    // compute euler step
    phi[gi] = phi[gi] + c_IT*dJ[gi]; 
}