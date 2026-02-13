/* phi density integration kernel */
__global__ void CuKernelInte(double* phi, double* psi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sum = 0.0;
        for(int k = 0; k < N; k++) sum += phi[k * N + i];
        psi[i] = sum;
    }
}

/* flux calculation kernel (interface at i + 1/2) */
__global__ void CuKernelComputeFlux(double *phi, double *J, double* percoll, double *R, double* I, double* gradWing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gi = i + j * N;

    if (i < N - 1) {
        auto get_v = [&](int idx) {

            double rp = (idx > wingL && idx < N - 1 - wingL) ? (R[j] + percoll[idx] - P0) 
                                                             : (R[j] + gradWing[idx] - P0);
            return (-c_alpha * rp - c_beta * I[idx]);
        };

        double v_face = 0.5 * (get_v(i) + get_v(i+1));
        
        double phi_up = (v_face > 0.0) ? phi[gi] : phi[gi+1];
        
         J[gi] = v_face * phi_up;

    } else if (i == N - 1) {
 
        J[gi] = 0.0; 
    }
}

__global__ void CuKernelUpdatePhi(double *phi, double *J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gi = i + j * N;

     if (i < N) {
        double flux_in = (i == 0) ? 0.0 : J[gi - 1];
        double flux_out = J[gi];
        double net_flux = (flux_in - flux_out) / c_IZ;
        
        phi[gi] = phi[gi] + c_IT * net_flux;
        
        if(phi[gi] < 0.0) phi[gi] = 0.0; 
    }
}

__global__ void CuKernelCmpA(double* y, double* alp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 1 && i < N - 1) alp[i] = 3.0 * (y[i+1] - y[i]) - 3.0 * (y[i] - y[i-1]);
}

__global__ void CuKernelSplineCoeffs(double* y, double* alp, double* b, double* c, double* d) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double mu[N], ze[N];
        mu[0] = 0.0; ze[0] = 0.0;
        for (int i = 1; i < N - 1; ++i) {
            mu[i] = 1.0 / (4.0 - mu[i-1]);
            ze[i] = (alp[i] - ze[i-1]) / (4.0 - mu[i-1]);
        }
        double localC[N], localB[N], localD[N];
        localC[N-1] = 0.0;
        for (int j = N - 2; j >= 0; --j) {
            localC[j] = ze[j] - mu[j] * localC[j + 1];
            localB[j] = (y[j+1] - y[j]) - (localC[j+1] + 2.0 * localC[j]) / 3.0;
            localD[j] = (localC[j+1] - localC[j]) / 3.0;
        }
        for (int idx = 0; idx < N; ++idx) {
            b[idx] = localB[idx]; c[idx] = localC[idx]; d[idx] = localD[idx];
        }
    }
}

__global__ void CuKernelSplineEval(double* y, double* b, double* c, double* d, double* psiIntp) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < M) {
        double x = (double)k / subDiv;
        int j = (int)floor(x);
        double dx = x - (double)j;
        psiIntp[k] = y[j] + (b[j] + (c[j] + d[j] * dx) * dx) * dx;
    }
}

__global__ void CuKernelConv(double* psi, double* I, double* convKernel) {
    extern __shared__ double s_psi[];
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int halo = (kernelN - 1) / 2;
    int blockStart = blockIdx.x * blockDim.x;

    if (gid < M) s_psi[tid + halo] = psi[gid];
    else s_psi[tid + halo] = 0.0;

    if (tid < halo) {
        int leftIdx = blockStart + tid - halo;
        s_psi[tid] = (leftIdx >= 0 ? psi[leftIdx] : 0.0);
    }
    if (tid >= blockDim.x - halo) {
        int offset = tid - (blockDim.x - halo);
        int rightIdx = blockStart + tid + halo;
        int sIdx = halo + blockDim.x + offset;
        s_psi[sIdx] = (rightIdx < M ? psi[rightIdx] : 0.0);
    }
    __syncthreads();

    if (gid < M) {
        double acc = 0.0;
        for (int k = 0; k < kernelN; ++k) acc += s_psi[tid + k] * c_convKernel[k];
        I[gid] = acc * (c_IZ / subDiv);
    }
}

__global__ void CuKernelDSmp(double* IIntp, double* I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i * subDiv;
    if (i < N) I[i] = IIntp[j];
}