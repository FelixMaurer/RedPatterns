import numpy as np

# =============================================================================
# USER DEFINED PARAMETERS
# (These were global variables in the C++ script)
# =============================================================================
kernelN = 31       # Length of the kernel (integer). Often an odd number.
IZ      = 1.041412353515625e-6      # Increment size dz (matching fine simulation grid)
subDiv  = 256        # Subdivision factor
U       = 100e-18       # Potential strength parameter for fLJ

# =============================================================================
# KERNEL GENERATION SCRIPT
# =============================================================================

def fLJ(r, sigma, U_param):
    """Lennard-Jones effective force."""
    return 4 * U_param * (12 * (sigma**12) / (r**13) - 6 * (sigma**6) / (r**7))

def g(r, d, sigmaC):
    """Gaussian PDF term."""
    return 4e7 * np.exp(-((r - d)**2) / (2 * (sigmaC**2)))

def generate_and_save_kernel():
    # 1. Setup local constants
    kernelL = (kernelN - 1) * IZ / subDiv
    kernelDZ = kernelL / (kernelN - 1)
    subRes = 10000
    # Equivalent to C++ integer division: (kernelN+1)/2
    half_kernel_n = (kernelN + 1) // 2 
    fineRes = int(subRes * half_kernel_n)
    
    fineDR = kernelDZ / subRes
    sigma = 5.6e-6
    sigmaC = 0.5e-6
    eqDist = 6.585467201064237091254725819933213415424688719213008880615234375e-06
    
    # 2. Compute fineR arrays and vectorized force/gpdf 
    # (starts at 1 to fineRes-1 just like the C++ loop: for(int i=1; i<fineRes; i++))
    fineR = np.arange(1, fineRes, dtype=np.float64) * fineDR
    
    force = fLJ(fineR, sigma, U)
    gpdf = g(fineR, eqDist, sigmaC)
    
    # Make up for numerical error near divergence
    gpdf[fineR < 1e-8] = 0.0
    
    # Calculate integration steps
    integrand = fineDR * force * gpdf
    
    # 3. Cumulative sum to simulate the C++ running `sum`
    kernelFine = np.zeros(fineRes, dtype=np.float64)
    # The C++ loop assigns the *previous* sum before adding the current integrand.
    # Therefore kernelFine[1] is 0, kernelFine[2] is integrand[0], etc.
    kernelFine[1:] = np.concatenate(([0.0], np.cumsum(integrand)[:-1]))
    
    # 4. Integration constant subtraction
    kernelFine = kernelFine[-1] - kernelFine
    
    # 5. Sampling of kernel
    intKernel = np.zeros(kernelN, dtype=np.float64)
    
    # C++ logic explicitly zeroes the middle/offset index
    intKernel[half_kernel_n] = 0.0
    
    for i in range(half_kernel_n, kernelN):
        kernelZ = (i * kernelDZ) - (kernelL / 2.0)
        # Note: using float division to mimic `double(kernelN+1)/2` in C++
        idx = int((i + 1 - (kernelN + 1) / 2.0) * subRes)
        
        intKernel[i] = kernelZ * kernelFine[idx]
        intKernel[kernelN - 1 - i] = -intKernel[i]
        
    print(f"kernel length = {kernelL:.32e} m")
    
    # 6. Save to disk (tab-separated single row to match C++ saveNVecToDrive)
    out_filename = "kernelInput.dat"
    # reshape(1, -1) ensures it prints as a single row instead of a single column
    np.savetxt(out_filename, intKernel.reshape(1, -1), delimiter='\t', fmt='%.18e')
    print(f"Kernel successfully saved to {out_filename}")

if __name__ == "__main__":
    generate_and_save_kernel()