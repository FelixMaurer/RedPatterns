/* phi density integration kernel */
__global__ void CuKernelInte(double* phi, double* psi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c_N) {
        double sum = 0.0;
        for(int k = 0; k < c_N; k++) sum += phi[k * c_N + i];
        psi[i] = sum;
    }
}

/* flux calculation kernel */
// J: Axial Flux (Z) at interface i+1/2
// K: Radial Flux (Rho) at interface j+1/2
__global__ void CuKernelComputeFlux(double *phi, double *J, double *K, double* percoll, double *R, double* I, double* gradWing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gi = i + j * c_N;

    // --- 1. Compute Axial Flux J (at i + 1/2) ---
    // Only valid for i < N-1. Boundary at i=N-1 is J=0.
    if (i < c_N - 1) {
        // Advection
        auto get_v = [&](int idx) {
            double rp = (idx > c_wingL && idx < c_N - 1 - c_wingL) ? (R[j] + percoll[idx] - P0) 
                                                             : (R[j] + gradWing[idx] - P0);
            return (-c_alpha * rp - c_beta * I[idx]);
        };
        double v_face = 0.5 * (get_v(i) + get_v(i+1));
        double phi_up = (v_face > 0.0) ? phi[gi] : phi[gi+1];
        
        // Diffusion (Z): Central difference at face
        // J_diff = -D * d_phi/dz
        double diff_flux_z = -c_D * (phi[gi+1] - phi[gi]) / c_IZ;

        J[gi] = v_face * phi_up + diff_flux_z;

    } else if (i == c_N - 1) {
        J[gi] = 0.0; 
    }

    // --- 2. Compute Radial Flux K (at j + 1/2) ---
    // Only valid for j < N-1. Boundary at j=N-1 is K=0.
    if (j < c_N - 1 && i < c_N) {
        // Diffusion (R) only
        // K_diff = -D * d_phi/dr
        // Note: phi is stored as phi[i + j*N], so next row is +N
        double diff_flux_r = -c_D * (phi[gi + c_N] - phi[gi]) / c_IR;
        
        K[gi] = diff_flux_r;

    } else if (j == c_N - 1 && i < c_N) {
        K[gi] = 0.0;
    }
}

__global__ void CuKernelUpdatePhi(double *phi, double *J, double *K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gi = i + j * c_N;

     if (i < c_N && j < c_N) {
        // --- Divergence in Z ---
        double flux_in_z  = (i == 0) ? 0.0 : J[gi - 1];
        double flux_out_z = J[gi];
        double div_z      = (flux_out_z - flux_in_z) / c_IZ; 

        // --- Divergence in R ---
        double flux_in_r  = (j == 0) ? 0.0 : K[gi - c_N];
        double flux_out_r = K[gi];
        double div_r      = (flux_out_r - flux_in_r) / c_IR;

        // Update: phi_new = phi_old - dt * (div_z + div_r)
        phi[gi] = phi[gi] - c_IT * (div_z + div_r);
        
        if(phi[gi] < 0.0) phi[gi] = 0.0; 
    }
}

__global__ void CuKernelCmpA(double* y, double* alp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 1 && i < c_N - 1) alp[i] = 3.0 * (y[i+1] - y[i]) - 3.0 * (y[i] - y[i-1]);
}

__global__ void CuKernelSplineCoeffs(double* y, double* alp, double* b, double* c, double* d, double* scratch) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Map scratch memory to arrays
        double* mu = scratch;
        double* ze = scratch + c_N;
        double* localC = scratch + 2 * c_N;
        double* localB = scratch + 3 * c_N;
        double* localD = scratch + 4 * c_N;

        mu[0] = 0.0; ze[0] = 0.0;
        for (int i = 1; i < c_N - 1; ++i) {
            mu[i] = 1.0 / (4.0 - mu[i-1]);
            ze[i] = (alp[i] - ze[i-1]) / (4.0 - mu[i-1]);
        }
        
        localC[c_N-1] = 0.0;
        for (int j = c_N - 2; j >= 0; --j) {
            localC[j] = ze[j] - mu[j] * localC[j + 1];
            localB[j] = (y[j+1] - y[j]) - (localC[j+1] + 2.0 * localC[j]) / 3.0;
            localD[j] = (localC[j+1] - localC[j]) / 3.0;
        }
        for (int idx = 0; idx < c_N; ++idx) {
            b[idx] = localB[idx]; c[idx] = localC[idx]; d[idx] = localD[idx];
        }
    }
}

__global__ void CuKernelSplineEval(double* y, double* b, double* c, double* d, double* psiIntp) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < c_M) {
        double x = (double)k / c_subDiv;
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

    if (gid < c_M) s_psi[tid + halo] = psi[gid];
    else s_psi[tid + halo] = 0.0;

    if (tid < halo) {
        int leftIdx = blockStart + tid - halo;
        s_psi[tid] = (leftIdx >= 0 ? psi[leftIdx] : 0.0);
    }
    if (tid >= blockDim.x - halo) {
        int offset = tid - (blockDim.x - halo);
        int rightIdx = blockStart + tid + halo;
        int sIdx = halo + blockDim.x + offset;
        s_psi[sIdx] = (rightIdx < c_M ? psi[rightIdx] : 0.0);
    }
    __syncthreads();

    if (gid < c_M) {
        double acc = 0.0;
        for (int k = 0; k < kernelN; ++k) acc += s_psi[tid + k] * c_convKernel[k];
        I[gid] = acc * (c_IZ / c_subDiv);
    }
}

__global__ void CuKernelDSmp(double* IIntp, double* I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (int)(i * c_subDiv); 
    if (i < c_N) I[i] = IIntp[j];
}