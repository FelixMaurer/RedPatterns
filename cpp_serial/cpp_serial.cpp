/*
    CPU Serial Implementation of Conservative Finite Volume Transport
    
    Features:
    - Single-threaded (Serial)
    - Auto-pins to CPU Core 0
    - Flushes Denormal numbers (FTZ)
    - Static geometry hoisted out of loop
    - Cache-friendly memory access in CpuInte
    - Debug output: "step_times.csv"
    
    Compile with: g++ -O3 -std=c++17 -ffast-math simulation_serial.cpp -o sim_serial
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

// Linux headers
#include <sched.h>
#include <pthread.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

namespace fs = std::filesystem;

// --- Macros ---
#define DISABLE_DENORMALS() \
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); \
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)

#define RC 1100.0 
#define RL 30.0 
#define kernelN 31 
#define mDeg 500
#define Rsigma 2.0f 
#define Rmu 1100.0f 
#define gradL 0.06 
#define P0 1100.0 
#define PL 8.0 

// --- Global Variables ---
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

double h_beta = 7.4e23; 
double h_alpha = 12e-5; 
double intKernel[kernelN]; 
double c_convKernel[kernelN]; 

// =========================================================
// HELPERS
// =========================================================

void pinToCore(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (result != 0) std::cerr << "Warning: Failed to pin thread.\n";
    else std::cout << "  [System] Pinned to Core " << core_id << "\n";
}

std::string createSimulationDirectory() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "sim_serial_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string dirName = ss.str();
    if (!fs::exists(dirName)) fs::create_directory(dirName);
    return dirName + "/";
}

void saveStepTimes(const std::vector<double>& times, std::string dir) {
    std::ofstream ofs(dir + "step_times.csv");
    ofs << "Step,Time_ms\n";
    for(size_t i = 0; i < times.size(); i++) {
        ofs << i << "," << times[i] << "\n";
    }
    ofs.close();
    std::cout << "  [Debug] Step times saved to " << dir << "step_times.csv\n";
}

void printHeader(int n_grid, int m_grid, int nt_steps, double dz_val, double dr_val, double dt_val, double psi_val, double isf_val, double d_val) {
    std::cout << "\n============================================================\n";
    std::cout << "   CPU Serial simulation: conservative finite volume transport\n";
    std::cout << "============================================================\n";
    std::cout << "  grid size (n)   : " << n_grid << "\n";
    std::cout << "  time steps (nt) : " << nt_steps << "\n";
    std::cout << "  Threads         : 1 (Pinned)\n";
    std::cout << "------------------------------------------------------------\n";
}

void printProgressBar(int step, int total, double elapsed_sec) {
    const int barWidth = 40;
    float progress = (float)step / total;
    std::cout << "\r[" << std::string(int(barWidth * progress), '=') << ">" 
              << std::string(barWidth - int(barWidth * progress), ' ') << "] " 
              << int(progress * 100.0) << " % " << std::flush;
}

bool loadNVecFromDrive(double* f, const char* inFileName, int n) {
    std::ifstream ifs(inFileName);
    if(!ifs.is_open()) return false;
    int count = 0;
    double v;
    while(count < n && (ifs >> v)) f[count++] = v;
    return (count == n);
}

void saveArrToDrive(double* f, std::string dir, std::string fileName) {
    const uint16_t sampleSkip = ceil(double(N)/256.0f);
    std::ofstream ofs(dir + fileName);
    for(int i=0; i<N; i+=sampleSkip){
        for(int j=0; j<N; j+=sampleSkip){
            if(j>0) ofs << "\t";
            ofs << f[i+N*j]; 
        }
        ofs << "\n";
    }
}

void saveVecToDrive(double* f, std::string dir, std::string fileName) {
    const uint16_t sampleSkip = ceil(double(N)/256.0f);
    std::ofstream ofs(dir + fileName);
    for(int i=0; i<N; i+=sampleSkip){
        if(i>0) ofs << "\t";
        ofs << f[i]; 
    }
    ofs << "\n";
}

// =========================================================
// PHYSICS KERNELS
// =========================================================

void initPhi(double *f) {
    double edgeZ = wingL + 2;
    double edgeR = wingL;
    double deltaI = IZ;      
    double deltaJ = IR;      
    
    // Gaussian params...
    double scaleSigmaZ = 0.6; double scaleSigmaR = 0.7;
    double posZ1 = 0.25; double posR1 = 0.45;
    double posZ2 = 0.75; double posR2 = 0.55;

    double totalLengthZ = (N - 1) * deltaI;
    double totalLengthR = (N - 1) * deltaJ;
    double baseSigmaIdx = Rsigma / deltaJ;
    double denZ = 2.0 * pow(baseSigmaIdx * scaleSigmaZ, 2);
    double denR = 2.0 * pow(baseSigmaIdx * scaleSigmaR, 2);

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double zVal = i * deltaI;
            double rVal = j * deltaJ;

            double dz1 = (zVal - totalLengthZ * posZ1) / deltaI;
            double dr1 = (rVal - totalLengthR * posR1) / deltaJ;
            double g1 = exp(-(dz1*dz1/denZ + dr1*dr1/denR));

            double dz2 = (zVal - totalLengthZ * posZ2) / deltaI;
            double dr2 = (rVal - totalLengthR * posR2) / deltaJ;
            double g2 = exp(-(dz2*dz2/denZ + dr2*dr2/denR));

            f[i + N * j] = g1 + g2;

            if((i < edgeZ) || (i > (N - 1 - edgeZ)) || (j < edgeR) || (j > (N - 1 - edgeR))) {
                f[i + N * j] = 0.0;
            }
        }
    }

    double phiSum = 0.0;
    for(int k = 0; k < N*N; k++) phiSum += f[k];
    if (phiSum > 0) {
        double normFactor = PSI * (N - 2 * edgeZ);
        for(int k = 0; k < N*N; k++) f[k] *= (normFactor / phiSum);
    }
}

void readParameters(int argc, char *argv[]){
    if(argc > 1) N = std::stoi(argv[1]);
    else exit(1);
    
    subDiv = 65536.0 / (double)N;
    M = int(N * subDiv + 1);
    wingL = (int)(N * (34.0 / 256.0));
    SYS_L = M * 1.041412353515625e-6;
    zShift = (SYS_L - gradL) / 2.0;

    IZ = SYS_L / (N - 1); 
    IR = RL / (N - 1);    

    if(argc>2) ISF = std::stod(argv[2]); 
    if(argc>3) PSI = std::stod(argv[3]); 
    if(argc>4) IT = std::stod(argv[4]); 
    if(argc>5) T = std::stod(argv[5]); 
    if(argc>6) NO = std::stod(argv[6]); 
    if(argc>7) D_coeff = std::stod(argv[7]); 
    NT = ceil(T/IT);
}

// --- OPTIMIZED CpuInte: Row-Major Access ---
// This reads memory sequentially (stride-1) instead of jumping by N (stride-N)
void CpuInte(const double* phi, double* psi) {
    // 1. Reset psi accumulator
    std::fill(psi, psi + N, 0.0);

    // 2. Loop row-by-row (j is row index)
    for (int j = 0; j < N; j++) {
        int rowOffset = j * N;
        // Inner loop reads phi[rowOffset + i] which is contiguous in memory
        for (int i = 0; i < N; i++) {
            psi[i] += phi[rowOffset + i];
        }
    }
}

void CpuCmpA(const double* y, double* alp) {
    for(int i = 1; i < N - 1; i++) {
        alp[i] = 3.0 * (y[i+1] - y[i]) - 3.0 * (y[i] - y[i-1]);
    }
}

void CpuSplineCoeffs(const double* y, const double* alp, double* b, double* c, double* d, double* mu, double* ze) {
    mu[0] = 0.0; ze[0] = 0.0;
    for (int i = 1; i < N - 1; ++i) {
        double den = 4.0 - mu[i-1];
        mu[i] = 1.0 / den;
        ze[i] = (alp[i] - ze[i-1]) / den;
    }
    c[N-1] = 0.0;
    for (int j = N - 2; j >= 0; --j) {
        c[j] = ze[j] - mu[j] * c[j + 1];
        b[j] = (y[j+1] - y[j]) - (c[j+1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j+1] - c[j]) / 3.0;
    }
}

void CpuSplineEval(const double* y, const double* b, const double* c, const double* d, double* psiIntp) {
    for (int k = 0; k < M; k++) {
        double x = (double)k / subDiv;
        int j = (int)floor(x);
        if (j >= N-1) j = N-2; 
        double dx = x - (double)j;
        psiIntp[k] = y[j] + (b[j] + (c[j] + d[j] * dx) * dx) * dx;
    }
}

void CpuConv(const double* psi, double* I, const double* kernel) {
    int halo = (kernelN - 1) / 2;
    for (int gid = 0; gid < M; gid++) {
        double acc = 0.0;
        for (int k = 0; k < kernelN; ++k) {
            int inputIdx = gid - halo + k;
            if (inputIdx >= 0 && inputIdx < M) acc += psi[inputIdx] * kernel[k];
        }
        I[gid] = acc * (IZ / subDiv);
    }
}

void CpuDSmp(const double* IIntp, double* I) {
    for (int i = 0; i < N; i++) {
        int j = (int)(i * subDiv);
        if (j < M) I[i] = IIntp[j];
    }
}

void CpuGrad(double* percoll) {
    for (int i = 0; i < N; i++) {
        double x = IZ * double(i);
        percoll[i] = (x - zShift - gradL/2) / (gradL/2) * PL / 2;
    }
}

void CpuWing(const double* percoll, double* gradWing) {
    double r3 = (percoll[wingL] - percoll[wingL-1]);
    double r2 = percoll[wingL];
    double r1 = r2 - 10;
    double x1 = 20;
    double x2 = wingL;
    
    if(percoll[(int)x1] < r1) r1 = percoll[(int)x1];

    double a = (r1 - r2 + r3 * (x2 - x1)) / ((x1 - x2) * (x1 - x2));
    double b_param = r3 - 2 * a * x2;
    double c_param = r2 - r3 * x2 + x2 * x2 * a;

    for (int i = 0; i < N; i++) {
        gradWing[i] = 0.0;
        if(i <= wingL) gradWing[i] = a*i*i + b_param * i + c_param;
        if(i >= N - 1 - wingL) gradWing[i] = -(a*(N-1-i)*(N-1-i) + b_param * (N-1-i) + c_param);
        
        if(i >= N - 1 - 13) gradWing[i] = gradWing[N - 1 - 13];
        if(i <= 13) gradWing[i] = gradWing[13];
    }
}

void CpuComputeFlux(const double* phi, double* J, double* K, const double* percoll, const double* R, const double* I, const double* gradWing) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int gi = i + j * N; 
            if (i < N - 1) {
                auto get_v = [&](int idx) {
                    double rp = (idx > wingL && idx < N - 1 - wingL) ? (R[j] + percoll[idx] - P0) 
                                                                     : (R[j] + gradWing[idx] - P0);
                    return (-h_alpha * rp - h_beta * I[idx]);
                };
                double v_face = 0.5 * (get_v(i) + get_v(i+1));
                double phi_up = (v_face > 0.0) ? phi[gi] : phi[gi+1];
                J[gi] = v_face * phi_up - D_coeff * (phi[gi+1] - phi[gi]) / IZ;
            } else J[gi] = 0.0;

            if (j < N - 1) {
                K[gi] = -D_coeff * (phi[gi + N] - phi[gi]) / IR;
            } else K[gi] = 0.0;
        }
    }
}

void CpuUpdatePhi(double* phi, const double* J, const double* K) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int gi = i + j * N;
            double flux_in_z  = (i == 0) ? 0.0 : J[gi - 1];
            double div_z      = (J[gi] - flux_in_z) / IZ;

            double flux_in_r  = (j == 0) ? 0.0 : K[gi - N];
            double div_r      = (K[gi] - flux_in_r) / IR;

            phi[gi] = phi[gi] - IT * (div_z + div_r);
            if(phi[gi] < 0.0) phi[gi] = 0.0;
        }
    }
}

// =========================================================
// MAIN
// =========================================================

void runSim() {
    std::string outDir = createSimulationDirectory();
    printHeader(N, M, NT, IZ, IR, IT, PSI, ISF, D_coeff);

    std::vector<double> R(N);
    for(int j=0; j<N; j++) R[j] = RC - RL/2 + RL * (double(j)/double(N-1));

    std::vector<double> phi(N * N);
    std::vector<double> J(N * N, 0.0), K(N * N, 0.0), I(N, 0.0);
    
    initPhi(phi.data());

    if(!loadNVecFromDrive(intKernel, "kernelInput.dat", kernelN)) exit(1);
    for(int i=0; i<kernelN; i++) intKernel[i] *= ISF;
    std::copy(std::begin(intKernel), std::end(intKernel), std::begin(c_convKernel));

    std::vector<double> psi(N, 0.0), alp(N, 0.0);
    std::vector<double> b(N, 0.0), c(N, 0.0), d(N, 0.0);
    std::vector<double> percoll(N, 0.0), gradWing(N, 0.0);
    std::vector<double> psiIntp(M, 0.0), IIntp(M, 0.0);

    struct WorkSpace {
        std::vector<double> mu, ze; 
        WorkSpace(int n) : mu(n), ze(n) {}
    } ws(N);

    // --- STATIC GEOMETRY (Calculated Once) ---
    CpuGrad(percoll.data());
    CpuWing(percoll.data(), gradWing.data());

    // --- TIMING DATA ---
    std::vector<double> step_times;
    step_times.reserve(NT); 

    std::cout << "  -> Starting simulation...\n";
    auto sim_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NT; i++) {
        // --- MEASURE START ---
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. Column Integration (Now Row-Major optimized)
        CpuInte(phi.data(), psi.data());
        
        // 2. Spline & Convolution
        CpuCmpA(psi.data(), alp.data());
        CpuSplineCoeffs(psi.data(), alp.data(), b.data(), c.data(), d.data(), ws.mu.data(), ws.ze.data());
        CpuSplineEval(psi.data(), b.data(), c.data(), d.data(), psiIntp.data());
        CpuConv(psiIntp.data(), IIntp.data(), c_convKernel);
        CpuDSmp(IIntp.data(), I.data());
        
        // 3. Flux & Update
        CpuComputeFlux(phi.data(), J.data(), K.data(), percoll.data(), R.data(), I.data(), gradWing.data());
        CpuUpdatePhi(phi.data(), J.data(), K.data());

        auto t_end = std::chrono::high_resolution_clock::now();
        // --- MEASURE END ---

        // Record time (Microseconds for better precision)
        std::chrono::duration<double, std::milli> diff = t_end - t_start;
        step_times.push_back(diff.count());

        // Output & Progress
        bool is_output = (((i - 1) % (int)NO) == 0) || (i == 1) || (i == (int)NT - 1);
        if (is_output) {
            std::chrono::duration<double> elapsed = t_end - sim_start_time;
            printProgressBar(i, NT, elapsed.count());

            std::stringstream ss;
            ss << std::setw(7) << std::setfill('0') << i; 
            saveArrToDrive(phi.data(), outDir, "phi_" + ss.str() + ".dat");
            saveVecToDrive(psi.data(), outDir, "psi_" + ss.str() + ".dat");
        }
    }

    // --- SAVE DEBUG FILE ---
    saveStepTimes(step_times, outDir);

    // --- FINAL STATS (With Outlier Filtering) ---
    if(!step_times.empty()) {
        // 1. Calculate Raw Mean (including spikes)
        // We skip the first 20 steps (warmup) to be fair
        size_t start_idx = (step_times.size() > 20) ? 20 : 0;
        size_t raw_count = step_times.size() - start_idx;
        
        if (raw_count > 0) {
            double raw_sum = 0.0;
            for(size_t k = start_idx; k < step_times.size(); k++) raw_sum += step_times[k];
            double raw_mean = raw_sum / raw_count;

            // 2. Filter Outliers (> 2.0 * raw_mean)
            double cutoff_threshold = 2.0 * raw_mean;
            std::vector<double> clean_times;
            clean_times.reserve(raw_count);
            
            int outliers = 0;
            for(size_t k = start_idx; k < step_times.size(); k++) {
                if(step_times[k] <= cutoff_threshold) {
                    clean_times.push_back(step_times[k]);
                } else {
                    outliers++;
                }
            }

            // 3. Calculate Clean Statistics
            double clean_sum = std::accumulate(clean_times.begin(), clean_times.end(), 0.0);
            double clean_mean = clean_times.empty() ? 0.0 : (clean_sum / clean_times.size());
            
            double clean_sq_sum = 0.0;
            for(double t : clean_times) {
                clean_sq_sum += (t - clean_mean) * (t - clean_mean);
            }
            double clean_stdev = clean_times.empty() ? 0.0 : std::sqrt(clean_sq_sum / clean_times.size());

            // 4. Print Report
            std::cout << "\n\n================ PERFORMANCE REPORT ================\n";
            std::cout << "  Raw Mean    : " << std::fixed << std::setprecision(4) << raw_mean << " ms\n";
            std::cout << "  Threshold   : " << cutoff_threshold << " ms ( > 2x Mean)\n";
            std::cout << "  Outliers    : " << outliers << " removed\n";
            std::cout << "  --------------------------------------------\n";
            std::cout << "  Clean Mean  : " << clean_mean << " ms\n";
            std::cout << "  Clean StdDev: " << clean_stdev << " ms\n";
            std::cout << "  Jitter      : " << (clean_stdev / clean_mean * 100.0) << " %\n";
            std::cout << "====================================================\n";
        }
    }
    
    // Still save the raw data for MATLAB debugging
    saveStepTimes(step_times, outDir);
}

int main(int argc, char *argv[]) {
    DISABLE_DENORMALS(); 
    pinToCore(0);
    if (argc < 2) { std::cerr << "Usage: ./sim <N> ...\n"; return 1; }
    readParameters(argc, argv);
    runSim();
    return 0;
}