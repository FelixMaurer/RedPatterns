#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdint>

// =============================================================================
// DEFINITIONS FROM definitions.h
// =============================================================================
#define subDiv 256.0
#define N 256
#define M_size int(N * subDiv + 1)
#define sysL (M_size * 1.041412353515625e-6)
#define kernelN 31

// =============================================================================
// CONSTANTS FROM constants.cu
// =============================================================================
const double IZ = sysL / (N - 1);
double U = -100e-18; 
double intKernel[kernelN];

// =============================================================================
// MACROS FROM functions.cu
// =============================================================================
#define fLJ(r, sigma) (4 * U * (12 * pow(sigma, 12) / pow(r, 13) - 6 * pow(sigma, 6) / pow(r, 7)))
#define g_func(r, d, sigmaC) (4e7 * exp(-pow(r - d, 2) / (2 * pow(sigmaC, 2))))

// =============================================================================
// KERNEL GENERATION (MIRRORING functions.cu)
// =============================================================================
void genConvKernel() {
    // 1. Setup constants precisely as defined in functions.cu
    double kernelL = (double(kernelN) - 1) * IZ / subDiv;
    double kernelDZ = kernelL / double(kernelN - 1);
    double subRes = 10000;
    double fineRes = subRes * (double(kernelN + 1) / 2);
    
    double fineDR = kernelDZ / subRes;
    double sigma = 5.6e-6;
    double sigmaC = 0.5e-6;
    double eqDist = 6.585467201064237091254725819933213415424688719213008880615234375e-06;

    // 2. Integration loop
    // Note: fineRes is 160,000 for kernelN=31
    std::vector<double> kernelFine(int(fineRes)); 
    double sum = 0;
    kernelFine[0] = 0;

    for (int i = 1; i < int(fineRes); i++) {
        double fineR = double(i * fineDR);
        double force = fLJ(fineR, sigma);
        double gpdf = g_func(fineR, eqDist, sigmaC);
        
        if (fineR < 1e-8) gpdf = 0.0;
        
        kernelFine[i] = sum;
        sum = sum + fineDR * force * gpdf;
    }

    // 3. Integration constant subtraction
    for (int i = 0; i < int(fineRes); i++) {
        kernelFine[i] = kernelFine[int(fineRes) - 1] - kernelFine[i];
    }

    // 4. Sampling of kernel
    intKernel[(kernelN + 1) / 2] = 0;
    for (int i = (kernelN + 1) / 2; i < kernelN; i++) {
        double kernelZ = double(i * kernelDZ) - kernelL / 2;
        
        // Use exact index calculation from functions.cu
        int idx = int((i + 1 - double(kernelN + 1) / 2.0) * subRes);
        
        if (idx < int(fineRes)) {
            intKernel[i] = kernelZ * kernelFine[idx];
            intKernel[kernelN - 1 - i] = -intKernel[i];
        }
    }

    std::cout << "kernel length = " << std::scientific << std::setprecision(32) << kernelL << " m" << std::endl;
}

// =============================================================================
// OUTPUT (MIRRORING saveNVecToDrive)
// =============================================================================
void saveNVecToDrive(double* f, const char* outFileName, int n) {
    std::ofstream ofs(outFileName);
    for (int i = 0; i < n; i++) {
        if (i > 0) ofs << "\t";
        ofs << std::scientific << std::setprecision(18) << f[i]; 
    }
    ofs << "\n";
    ofs.close();
}

int main() {
    genConvKernel();
    saveNVecToDrive(intKernel, "intKernel.dat", kernelN);
    std::cout << "Kernel successfully saved to intKernel.dat" << std::endl;
    return 0;
}