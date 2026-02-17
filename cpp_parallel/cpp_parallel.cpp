/*
    CPU Optimized Implementation of Conservative Finite Volume Transport
    Replaces CUDA kernels with OpenMP parallel loops.
    
    Compile with: g++ -O3 -std=c++17 -fopenmp simulation_cpu.cpp -o sim_cpu
*/

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <omp.h> // OpenMP for CPU parallelism

namespace fs = std::filesystem;

// =========================================================
// DEFINITIONS & CONSTANTS (from definitions.h & constants.cu)
// =========================================================

// --- Macros ---
#define RC 1100.0 
#define RL 30.0 
#define kernelN 31 
#define mDeg 500
#define Rsigma 2.0f 
#define Rmu 1100.0f 
#define gradL 0.06 
#define P0 1100.0 
#define PI 3.141592653589793115997963468544185161590576171875
#define PL 8.0 // For linear gradient

// --- Global Variables ---
// (Formerly externs or managed in constants.cu)
double SYS_L; 
double subDiv; 
int M; 
int wingL;
double zShift;

double PSI = 0.02; 
double ISF = 1.000;
double IT = 0.005;   
double T = 1200.0; 
double NO = 1000; 
double D_coeff = 0.0; 
int N;         
uint32_t NT; 
double IZ; 
double IR; 

// Host flux prefactors
double h_beta = 7.4e23; 
double h_alpha = 12e-5; 

// Kernel data
double intKernel[kernelN]; 
double c_convKernel[kernelN]; // CPU copy of kernel

// =========================================================
// HELPER FUNCTIONS (I/O)
// =========================================================

std::string createSimulationDirectory() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "sim_cpu_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string dirName = ss.str();
    if (!fs::exists(dirName)) fs::create_directory(dirName);
    return dirName + "/";
}

void printHeader(int n_grid, int m_grid, int nt_steps, double dz_val, double dr_val, double dt_val, double psi_val, double isf_val, double d_val) {
    std::cout << "\n============================================================\n";
    std::cout << "   CPU simulation: conservative finite volume transport     \n";
    std::cout << "============================================================\n";
    std::cout << "  grid size (n)   : " << n_grid << "\n";
    std::cout << "  subgrid (m)     : " << m_grid << "\n";
    std::cout << "  wing length     : " << wingL << "\n";
    std::cout << "  time steps (nt) : " << nt_steps << "\n";
    std::cout << "  dz (space step) : " << dz_val << "\n";
    std::cout << "  dr (dens. step) : " << dr_val << "\n";
    std::cout << "  dt (time step)  : " << dt_val << "\n";
    std::cout << "  density (psi)   : " << psi_val << "\n";
    std::cout << "  scale factor    : " << isf_val << "\n";
    std::cout << "  diffusion (D)   : " << d_val << "\n";
    std::cout << "  Threads (OMP)   : " << omp_get_max_threads() << "\n";
    std::cout << "------------------------------------------------------------\n";
}

void printProgressBar(int step, int total, double elapsed_sec) {
    const int barWidth = 40;
    float progress = (float)step / total;
    
    std::cout << "\r[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " % ";
    
    if (step > 0) {
        double eta_sec = (elapsed_sec / step) * (total - step);
        int h = (int)(eta_sec / 3600);
        int m = (int)((eta_sec - h*3600) / 60);
        int s = (int)eta_sec % 60;
        std::cout << "| eta: " 
                  << std::setfill('0') << std::setw(2) << h << ":"
                  << std::setfill('0') << std::setw(2) << m << ":"
                  << std::setfill('0') << std::setw(2) << s;
    }
    std::cout << std::flush;
}

void saveNVecToDrive(const double* f, std::string dir, std::string fileName, int n) {
    std::string fullPath = dir + fileName;
    std::ofstream ofs(fullPath);
    ofs.imbue(std::locale::classic());
    ofs << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);
    for(int i=0; i<n; i++){
        if(i>0) ofs << "\t";
        ofs << f[i]; 
    }
    ofs << "\n";
    ofs.close();
}

bool loadNVecFromDrive(double* f, const char* inFileName, int n) {
    std::ifstream ifs(inFileName);
    if(!ifs.is_open()){
        fprintf(stderr, "\nerror: could not open kernel file '%s'\n", inFileName);
        return false;
    }
    ifs.imbue(std::locale::classic());
    int count = 0;
    double v;
    while(count < n && (ifs >> v)) f[count++] = v;
    if(count < n){
        fprintf(stderr, "\nerror: kernel file '%s' has only %d values (need %d)\n", inFileName, count, n);
        return false;
    }
    return true;
}

void saveArrToDrive(double* f, std::string dir, std::string fileName) {
    const uint16_t sampleSkip = ceil(double(N)/256.0f);
    std::string fullPath = dir + fileName;
    std::ofstream ofs(fullPath);
    for(int i=0; i<N; i+=sampleSkip){
        for(int j=0; j<N; j+=sampleSkip){
            if(j>0) ofs << "\t";
            ofs << f[i+N*j]; 
        }
        ofs << "\n";
    }
    ofs.close();
}

void saveVecToDrive(double* f, std::string dir, std::string fileName) {
    const uint16_t sampleSkip = ceil(double(N)/256.0f);
    std::string fullPath = dir + fileName;
    std::ofstream ofs(fullPath);
    for(int i=0; i<N; i+=sampleSkip){
        if(i>0) ofs << "\t";
        ofs << f[i]; 
    }
    ofs << "\n";
    ofs.close();
}

void initPhi(double *f) {
    double edgeZ = wingL + 2;
    double edgeR = wingL;
    
    const double deltaI = IZ;      
    const double deltaJ = IR;      
    
    double scaleSigmaZ = 0.6; 
    double scaleSigmaR = 0.7;

    double posZ1 = 0.25; 
    double posR1 = 0.45;
    double posZ2 = 0.75; 
    double posR2 = 0.55;

    double totalLengthZ = (N - 1) * deltaI;
    double totalLengthR = (N - 1) * deltaJ;

    double zCenter1 = totalLengthZ * posZ1;
    double rCenter1 = totalLengthR * posR1;
    double zCenter2 = totalLengthZ * posZ2;
    double rCenter2 = totalLengthR * posR2;

    double baseSigmaIdx = Rsigma / deltaJ;
    double sIdxZ = baseSigmaIdx * scaleSigmaZ;
    double sIdxR = baseSigmaIdx * scaleSigmaR;

    double denZ = 2.0 * pow(sIdxZ, 2);
    double denR = 2.0 * pow(sIdxR, 2);

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double zVal = i * deltaI;
            double rVal = j * deltaJ;

            double dz1 = (zVal - zCenter1) / deltaI;
            double dr1 = (rVal - rCenter1) / deltaJ;
            double g1 = exp(-(dz1*dz1/denZ + dr1*dr1/denR));

            double dz2 = (zVal - zCenter2) / deltaI;
            double dr2 = (rVal - rCenter2) / deltaJ;
            double g2 = exp(-(dz2*dz2/denZ + dr2*dr2/denR));

            f[i + N * j] = g1 + g2;

            if((i < edgeZ) || (i > (N - 1 - edgeZ)) || (j < edgeR) || (j > (N - 1 - edgeR))) {
                f[i + N * j] = 0.0;
            }
        }
    }

    double phiSum = 0.0;
    // Parallel reduction for sum
    #pragma omp parallel for reduction(+:phiSum)
    for(int k = 0; k < N*N; k++) phiSum += f[k];

    if (phiSum > 0) {
        double normFactor = PSI * (N - 2 * edgeZ);
        #pragma omp parallel for
        for(int k = 0; k < N*N; k++) {
            f[k] *= (normFactor / phiSum);
        }
    }
}

void readParameters(int argc, char *argv[]){
    int argIdx = 1;
    if(argc > argIdx) {
        N = std::stoi(argv[argIdx]);
        argIdx++;
    } else {
        std::cerr << "Error: N (grid size) must be provided as the first argument." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    subDiv = 65536.0 / (double)N;
    M = int(N * subDiv + 1);
    wingL = (int)(N * (34.0 / 256.0));
    SYS_L = M * 1.041412353515625e-6;
    zShift = (SYS_L - gradL) / 2.0;

    IZ = SYS_L / (N - 1); 
    IR = RL / (N - 1);    

    if(argc>argIdx) ISF = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) PSI = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) IT = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) T = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) NO = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) D_coeff = std::stod(argv[argIdx]); argIdx++;
    NT = ceil(T/IT);
}

// =========================================================
// CPU "KERNELS"
// =========================================================

// Integrates phi to get psi (column sum)
// Note: phi is row-major [i + j*N] where i is Z (col), j is R (row)
// But access in CUDA Inte is phi[k*N + i], meaning sum over k (rows) for fixed i (col).
void CpuInte(const double* phi, double* psi) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for(int k = 0; k < N; k++) sum += phi[k * N + i];
        psi[i] = sum;
    }
}

// Compute 'alp' array (finite difference of psi)
void CpuCmpA(const double* y, double* alp) {
    #pragma omp parallel for
    for(int i = 1; i < N - 1; i++) {
        alp[i] = 3.0 * (y[i+1] - y[i]) - 3.0 * (y[i] - y[i-1]);
    }
    // Boundaries are left as 0 from initialization, matching CUDA
}

// Spline Coefficients (Tri-diagonal solver)
// This is inherently serial in the 'i' dimension, so no OMP parallel for logic here.
// Now accepts mu and ze as pointers to prevent per-step allocations
void CpuSplineCoeffs(const double* y, const double* alp, double* b, double* c, double* d, double* mu, double* ze) {
    mu[0] = 0.0; 
    ze[0] = 0.0;
    for (int i = 1; i < N - 1; ++i) {
        double den = 4.0 - mu[i-1];
        mu[i] = 1.0 / den;
        ze[i] = (alp[i] - ze[i-1]) / den;
    }
    
    c[N-1] = 0.0;
    for (int j = N - 2; j >= 0; --j) {
        c[j] = ze[j] - mu[j] * c[j + 1];
        double c_curr = c[j];
        double c_next = c[j+1];
        b[j] = (y[j+1] - y[j]) - (c_next + 2.0 * c_curr) / 3.0;
        d[j] = (c_next - c_curr) / 3.0;
    }
}

// Evaluate Spline on finer grid M
void CpuSplineEval(const double* y, const double* b, const double* c, const double* d, double* psiIntp) {
    #pragma omp parallel for
    for (int k = 0; k < M; k++) {
        double x = (double)k / subDiv;
        int j = (int)floor(x);
        // Clamp j to be within bounds (though x should be safe, float precision might vary)
        if (j >= N-1) j = N-2; 
        
        double dx = x - (double)j;
        psiIntp[k] = y[j] + (b[j] + (c[j] + d[j] * dx) * dx) * dx;
    }
}

// Convolution
void CpuConv(const double* psi, double* I, const double* kernel) {
    int halo = (kernelN - 1) / 2;
    
    #pragma omp parallel for
    for (int gid = 0; gid < M; gid++) {
        double acc = 0.0;
        for (int k = 0; k < kernelN; ++k) {
            int inputIdx = gid - halo + k;
            double val = 0.0;
            if (inputIdx >= 0 && inputIdx < M) {
                val = psi[inputIdx];
            }
            acc += val * kernel[k];
        }
        I[gid] = acc * (IZ / subDiv);
    }
}

// Downsample
void CpuDSmp(const double* IIntp, double* I) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        int j = (int)(i * subDiv);
        if (j < M) I[i] = IIntp[j];
    }
}

// Linear Gradient (replacing CuKernelGrad from cuda_kernel_linear.cu)
void CpuGrad(double* percoll) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double x = IZ * double(i);
        percoll[i] = (x - zShift - gradL/2) / (gradL/2) * PL / 2;
    }
}

// Wing Gradient (replacing CuKernelWing)
void CpuWing(const double* percoll, double* gradWing) {
    // We need to calculate the parabola params first. 
    // Since these depend on specific indices, we calculate them once serially or redundantly.
    // It's cheap, so let's do it inside the loop or pre-calc.
    // Pre-calc is safer to match CUDA logic which does it per thread (redundantly).
    
    double r3 = (percoll[wingL] - percoll[wingL-1]);
    double r2 = percoll[wingL];
    double r1 = r2 - 10;
    double x1 = 20;
    double x2 = wingL;
    
    if(percoll[(int)x1] < r1) r1 = percoll[(int)x1];

    double a = (r1 - r2 + r3 * (x2 - x1)) / ((x1 - x2) * (x1 - x2));
    double b_param = r3 - 2 * a * x2;
    double c_param = r2 - r3 * x2 + x2 * x2 * a;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        gradWing[i] = 0.0;
        if(i <= wingL)
            gradWing[i] = a*i*i + b_param * i + c_param;
        if(i >= N - 1 - wingL)
            gradWing[i] = -(a*(N-1-i)*(N-1-i) + b_param * (N-1-i) + c_param);
        
        // Specific hardcoded fixes from CUDA code
        if(i >= N - 1 - 13)
            gradWing[i] = gradWing[N - 1 - 13];
        if(i <= 13)
            gradWing[i] = gradWing[13];
    }
}

// Compute Fluxes J and K
void CpuComputeFlux(const double* phi, double* J, double* K, const double* percoll, const double* R, const double* I, const double* gradWing) {
    // J: Z-flux (N*N)
    // K: R-flux (N*N)
    
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int gi = i + j * N; // Global Index

            // --- 1. Axial Flux J (at i + 1/2) ---
            if (i < N - 1) {
                // Lambda for velocity calculation
                auto get_v = [&](int idx) {
                    double rp = (idx > wingL && idx < N - 1 - wingL) ? (R[j] + percoll[idx] - P0) 
                                                                     : (R[j] + gradWing[idx] - P0);
                    return (-h_alpha * rp - h_beta * I[idx]);
                };
                
                double v_face = 0.5 * (get_v(i) + get_v(i+1));
                double phi_up = (v_face > 0.0) ? phi[gi] : phi[gi+1];
                
                double diff_flux_z = -D_coeff * (phi[gi+1] - phi[gi]) / IZ;
                J[gi] = v_face * phi_up + diff_flux_z;
            } else {
                J[gi] = 0.0;
            }

            // --- 2. Radial Flux K (at j + 1/2) ---
            if (j < N - 1) {
                // Access phi[gi + N] which is phi at i, j+1
                double diff_flux_r = -D_coeff * (phi[gi + N] - phi[gi]) / IR;
                K[gi] = diff_flux_r;
            } else {
                K[gi] = 0.0;
            }
        }
    }
}

// Update Phi based on flux divergence
void CpuUpdatePhi(double* phi, const double* J, const double* K) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int gi = i + j * N;
            
            // Div Z
            double flux_in_z  = (i == 0) ? 0.0 : J[gi - 1];
            double flux_out_z = J[gi];
            double div_z      = (flux_out_z - flux_in_z) / IZ;

            // Div R
            double flux_in_r  = (j == 0) ? 0.0 : K[gi - N];
            double flux_out_r = K[gi];
            double div_r      = (flux_out_r - flux_in_r) / IR;

            phi[gi] = phi[gi] - IT * (div_z + div_r);
            if(phi[gi] < 0.0) phi[gi] = 0.0;
        }
    }
}

// =========================================================
// MAIN SIMULATION LOOP
// =========================================================

void runSim() {
    std::string outDir = createSimulationDirectory();
    printHeader(N, M, NT, IZ, IR, IT, PSI, ISF, D_coeff);

    // --- Allocations (Host Memory) ---
    std::vector<double> R(N);
    for(int j=0; j<N; j++) R[j] = RC - RL/2 + RL * (double(j)/double(N-1));

    std::vector<double> phi(N * N);
    std::vector<double> J(N * N, 0.0);
    std::vector<double> K(N * N, 0.0);
    std::vector<double> I(N, 0.0);
    
    initPhi(phi.data());

    // Kernel Loading
    if(!loadNVecFromDrive(intKernel, "kernelInput.dat", kernelN)){
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<kernelN; i++) intKernel[i] *= ISF;
    std::copy(std::begin(intKernel), std::end(intKernel), std::begin(c_convKernel));

    // Working Arrays
    std::vector<double> psi(N, 0.0), alp(N, 0.0);
    std::vector<double> b(N, 0.0), c(N, 0.0), d(N, 0.0);
    std::vector<double> percoll(N, 0.0), gradWing(N, 0.0);
    std::vector<double> psiIntp(M, 0.0), IIntp(M, 0.0);

    // NEW: Timing WorkSpace (Prevents std::dev spikes from OS allocator)
    struct WorkSpace {
        std::vector<double> mu, ze; 
        WorkSpace(int n) : mu(n), ze(n) {}
    } ws(N);

    // Initial Setup
    CpuGrad(percoll.data());
    CpuWing(percoll.data(), gradWing.data());

    // Timing Configuration
    std::vector<double> step_times;
    step_times.reserve(NT); 
    const int WARMUP = 20; 
    const int COOLDOWN = 20;
    double t_sim = 0.0;

    std::cout << "  -> Starting simulation loop...\n\n";
    auto sim_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NT; i++) {
        // Evaluate logic flags
        bool is_output_step = (((i - 1) % (int)NO) == 0) || (i == 1) || (i == (int)NT - 1);
        bool can_measure = (i >= WARMUP) && (i < (int)NT - COOLDOWN) && !is_output_step;

        // --- MEASURED COMPUTE BLOCK ---
        auto t_start = std::chrono::high_resolution_clock::now();

        CpuInte(phi.data(), psi.data());
        CpuCmpA(psi.data(), alp.data());
        
        // Pass the Workspace buffers
        CpuSplineCoeffs(psi.data(), alp.data(), b.data(), c.data(), d.data(), ws.mu.data(), ws.ze.data());
        
        CpuSplineEval(psi.data(), b.data(), c.data(), d.data(), psiIntp.data());
        CpuConv(psiIntp.data(), IIntp.data(), c_convKernel);
        CpuDSmp(IIntp.data(), I.data());
        
        CpuGrad(percoll.data());
        CpuWing(percoll.data(), gradWing.data());

        CpuComputeFlux(phi.data(), J.data(), K.data(), percoll.data(), R.data(), I.data(), gradWing.data());
        CpuUpdatePhi(phi.data(), J.data(), K.data());

        auto t_end = std::chrono::high_resolution_clock::now();
        // ------------------------------

        if (can_measure) {
            std::chrono::duration<double, std::milli> diff = t_end - t_start;
            step_times.push_back(diff.count());
        }

        if (is_output_step) {
            std::chrono::duration<double> elapsed = t_end - sim_start_time;
            printProgressBar(i, NT, elapsed.count());

            std::string s_idx = std::to_string(i);
            saveArrToDrive(phi.data(), outDir, "phi_" + s_idx + ".dat");
            saveVecToDrive(psi.data(), outDir, "psi_" + s_idx + ".dat");
        }
        t_sim += IT;
    }

    // --- FINAL STATS ---
    if(!step_times.empty()) {
        double sum = std::accumulate(step_times.begin(), step_times.end(), 0.0);
        double mean = sum / step_times.size();
        double sq_sum = 0.0;
        for(double val : step_times) sq_sum += (val - mean) * (val - mean);
        double stdev = std::sqrt(sq_sum / step_times.size());

        std::cout << "\n\n================ PERFORMANCE ================\n";
        std::cout << "  Avg Step: " << std::fixed << std::setprecision(4) << mean << " ms\n";
        std::cout << "  Std Dev : " << stdev << " ms (" << (stdev/mean)*100.0 << "% jitter)\n";
        std::cout << "=============================================\n";
    }
}

int main(int argc, char *argv[]) {
    // Check Args
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <GridSize_N> [ISF] [PSI] [IT] [T] [NO] [D]" << std::endl;
        return 1;
    }

    readParameters(argc, argv);
    runSim();
    
    return 0;
}