/*
    This contains all CUDA kernels.
    integration -> interpolation (CmpA) -> interpolation (CmpL)
    -> downsampling (DSmp) -> convolution (Conv) -> gradient (Grad)
    -> gradient wing (Wing) -> iteration (Iter)
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
        alp[i] = 3.0 * (y[i + 1] - y[i]) / 1.0 - 3.0 * (y[i] - y[i - 1]) / 1.0;
}
__global__ void CuKernelCmpL(double* y, double* alp, double* psiIntp)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    double mu[N], z[N];
    mu[0] = 0;
    z[0] = 0;
    for (int i = 1; i < N-1; ++i) {
        mu[i] = 1/(4.0-mu[i-1]);
        z[i] = (alp[i]-z[i-1])/(4.0-mu[i-1]);
    }
    z[N-1] = 0;
    mu[N-1] = 0;
    double d[N], b[N], c[N];
    c[N-1] = 0;
    for (int j = N-2; j>=0; --j) {
        c[j] = z[j] - mu[j] * c[j + 1];
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
            sum += psi[i+(k-d)]*convKernel[k]*(c_IX/subDiv);
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
        rpTerm = R[j]-percoll[i]-b1;
    __syncthreads();
    if((i<=wingL) | (i >= N-1-wingL))
        rpTerm = R[j]-gradWing[i]-b1;
    __syncthreads();
    J[gi] = (c_alpha*rpTerm + c_beta*I[i])*phi[gi];
    __syncthreads();
    // compute flux derivative
    if((i>=4) & (i<=N-1-4)){
        // physical flux first derivative
        dJ[gi] = (
            +  0.5/c_IX * ( J[gi+1] - J[gi-1] )
            );
        // degenerate diffusion second derivative
        dJ[gi] += (
            - 2.0 * ( jDegDiff(gi) )
            + 1.0 * ( jDegDiff(gi+1)+jDegDiff(gi-1) )
            )/(c_IX*c_IX);
        /* (optional) degenerate diffusion 1 second derivative
        dJ[gi] += (
            - 2.0 * ( jDegDiff1(i) )
            + 1.0 * ( jDegDiff1(i+1)+jDegDiff1(i-1) )
            )/(c_IX*c_IX);*/
    }
    __syncthreads();
    // compute euler step
    phi[gi] = phi[gi] - IT*dJ[gi]; 
}