import numpy as np

# =============================================================================
# CONSTANTS FROM OLD C++ SCRIPTS (functions.cu, constants.cu, definitions.h)
# =============================================================================
kernelN = 31
subDiv  = 256.0
# sysL is derived as (M * 1.041412353515625e-6) where M = int(N * subDiv + 1)
N_size  = 256
M_size  = int(N_size * subDiv + 1)
sysL    = M_size * 1.041412353515625e-6
IZ      = sysL / (N_size - 1)
U       = 100e-18 # Value usually passed via readParameters

def fLJ(r, sigma, U_param):
    """Lennard-Jones effective force: #define fLJ in functions.cu"""
    return 4 * U_param * (12 * (sigma**12) / (r**13) - 6 * (sigma**6) / (r**7))

def g(r, d, sigmaC):
    """Gaussian PDF term: #define g in functions.cu"""
    return 4e7 * np.exp(-((r - d)**2) / (2 * (sigmaC**2)))

def generate_and_save_kernel():
    # 1. Setup constants from genConvKernel()
    kernelL = (float(kernelN) - 1.0) * IZ / subDiv
    kernelDZ = kernelL / float(kernelN - 1)
    subRes = 10000.0
    
    # Matches: double fineRes = subRes*(double(kernelN+1)/2);
    fineRes = int(subRes * (float(kernelN + 1) / 2.0))
    
    fineDR = kernelDZ / subRes
    sigma = 5.6e-6
    sigmaC = 0.5e-6
    eqDist = 6.585467201064237091254725819933213415424688719213008880615234375e-06
    
    # 2. Compute effective potential (kernelFine)
    kernelFine = np.zeros(fineRes, dtype=np.float64)
    current_sum = 0.0
    kernelFine[0] = 0.0 # avoid divergence at zero
    
    # C++: for(int i=1; i<fineRes; i++)
    for i in range(1, fineRes):
        fineR = float(i * fineDR)
        force = fLJ(fineR, sigma, U)
        gpdf = g(fineR, eqDist, sigmaC)
        
        if fineR < 1e-8: # make up for numerical error near divergence
            gpdf = 0.0
            
        kernelFine[i] = current_sum # compute integral
        current_sum += fineDR * force * gpdf

    # 3. Integration constant
    # Matches: kernelFine[i] = kernelFine[int(fineRes)-1] - kernelFine[i];
    last_val = kernelFine[fineRes - 1]
    for i in range(fineRes):
        kernelFine[i] = last_val - kernelFine[i]
    
    # 4. Sampling of kernel
    intKernel = np.zeros(kernelN, dtype=np.float64)
    # Matches: intKernel[(kernelN+1)/2] = 0;
    mid_idx = (kernelN + 1) // 2
    intKernel[mid_idx] = 0.0
    
    # C++: for(int i=(kernelN+1)/2; i<kernelN; i++)
    for i in range(mid_idx, kernelN):
        kernelZ = float(i * kernelDZ) - (kernelL / 2.0)
        
        # Exact indexing from functions.cu: 
        # int((i+1-double(kernelN+1)/2.0)*subRes)
        # Added 1e-11 to prevent floating point noise from truncating incorrectly
        idx = int(((i + 1) - (float(kernelN + 1) / 2.0)) * subRes + 1e-11)
        
        if idx < fineRes:
            intKernel[i] = kernelZ * kernelFine[idx]
            # Matches: intKernel[kernelN-1-i] = -intKernel[i];
            intKernel[kernelN - 1 - i] = -intKernel[i]
        
    print(f"kernel length = {kernelL:.32e} m")
    
    # 5. Save to disk (tab-separated single row)
    # Matches: saveNVecToDrive logic
    out_filename = "intKernel.dat"
    np.savetxt(out_filename, intKernel.reshape(1, -1), delimiter='\t', fmt='%.18e')
    print(f"Kernel successfully saved to {out_filename}")

if __name__ == "__main__":

    generate_and_save_kernel()
