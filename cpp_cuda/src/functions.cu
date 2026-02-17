namespace fs = std::filesystem;
#include <vector>
#include <numeric>
#include <cmath>

void printHeader(int n_grid, int m_grid, int nt_steps, double dz_val, double dr_val, double dt_val, double psi_val, double isf_val, double d_val) {
    std::cout << "\n============================================================\n";
    std::cout << "   cuda simulation: conservative finite volume transport    \n";
    std::cout << "============================================================\n";
    std::cout << "  grid size (n)   : " << n_grid << "\n";
    std::cout << "  subgrid (m)     : " << m_grid << "\n";
    std::cout << "  wing length     : " << wingL << "\n";
    std::cout << "  time steps (nt) : " << nt_steps << "\n";
    std::cout << "  dz (space step) : " << dz_val << "\n";
    std::cout << "  dr (dens. step) : " << dr_val << "\n";
    std::cout << "  dt (time step)  : " << dt_val << "\n";
    std::cout << "  density (psi)   : " << psi_val << "\n";
    std::cout << "  scale factor (isf): " << isf_val << "\n";
    std::cout << "  diffusion (D)   : " << d_val << "\n";
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

std::string createSimulationDirectory() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "sim_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string dirName = ss.str();
    if (!fs::exists(dirName)) fs::create_directory(dirName);
    return dirName + "/";
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

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) fprintf(stderr, "\ncuda runtime error: %s\n", cudaGetErrorString(result));
    return result;
}

void initPhi(double *f, double *R) {
    double edgeZ = wingL + 2;
    double edgeR = wingL;
    
    // --- Configuration Parameters ---
    const double deltaI = IZ;      // meters (Z)
    const double deltaJ = IR;      // g/L (R)
    
    // Independent sigma scales (1.0 = original Rsigma-equivalent width)
    double scaleSigmaZ = 0.6; 
    double scaleSigmaR = 0.7;

    // --- Gaussian Positions (Relative 0.0 to 1.0) ---
    double posZ1 = 0.25; 
    double posR1 = 0.45;
    double posZ2 = 0.75; 
    double posR2 = 0.55;

    // Physical dimensions & Centers
    double totalLengthZ = (N - 1) * deltaI;
    double totalLengthR = (N - 1) * deltaJ;

    double zCenter1 = totalLengthZ * posZ1;
    double rCenter1 = totalLengthR * posR1;
    double zCenter2 = totalLengthZ * posZ2;
    double rCenter2 = totalLengthR * posR2;

    // Calculate separate sigmas in index-space
    double baseSigmaIdx = Rsigma / deltaJ;
    double sIdxZ = baseSigmaIdx * scaleSigmaZ;
    double sIdxR = baseSigmaIdx * scaleSigmaR;

    // Pre-calculate denominators for performance
    double denZ = 2.0 * pow(sIdxZ, 2);
    double denR = 2.0 * pow(sIdxR, 2);

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double zVal = i * deltaI;
            double rVal = j * deltaJ;

            // Gaussian 1
            double dz1 = (zVal - zCenter1) / deltaI;
            double dr1 = (rVal - rCenter1) / deltaJ;
            double g1 = exp(-(dz1*dz1/denZ + dr1*dr1/denR));

            // Gaussian 2
            double dz2 = (zVal - zCenter2) / deltaI;
            double dr2 = (rVal - rCenter2) / deltaJ;
            double g2 = exp(-(dz2*dz2/denZ + dr2*dr2/denR));

            f[i + N * j] = g1 + g2;

            if((i < edgeZ) || (i > (N - 1 - edgeZ)) || (j < edgeR) || (j > (N - 1 - edgeR))) {
                f[i + N * j] = 0.0;
            }
        }
    }

    // --- Normalization ---
    double phiSum = 0.0;
    for(int i = 0; i < N; i++) for(int j = 0; j < N; j++) phiSum += f[i + N * j];

    if (phiSum > 0) {
        double normFactor = PSI * (N - 2 * edgeZ);
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                f[i + N * j] *= (normFactor / phiSum);
            }
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

    // Calculate increments
    IZ = SYS_L / (N - 1); // Space (m)
    IR = RL / (N - 1);    // Density (g/L)

    if(argc>argIdx) ISF = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) PSI = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) IT = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) T = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) NO = std::stod(argv[argIdx]); argIdx++;
    if(argc>argIdx) D_coeff = std::stod(argv[argIdx]); argIdx++;
    NT = ceil(T/IT);
}

void runSim(){
    std::string outDir = createSimulationDirectory();
    
    printHeader(N, M, NT, IZ, IR, IT, PSI, ISF, D_coeff);
    
    std::cout << "  output dir      : " << outDir << "\n";
    std::cout << "============================================================\n\n";

    checkCuda( cudaMemcpyToSymbol(c_IZ, &IZ, sizeof(double)) );
    checkCuda( cudaMemcpyToSymbol(c_IR, &IR, sizeof(double)) );
    checkCuda( cudaMemcpyToSymbol(c_IT, &IT, sizeof(double)) );
    checkCuda( cudaMemcpyToSymbol(c_PSI, &PSI, sizeof(double)) );
    checkCuda( cudaMemcpyToSymbol(c_beta, &h_beta, sizeof(double)) );
    checkCuda( cudaMemcpyToSymbol(c_alpha, &h_alpha, sizeof(double)) );
    checkCuda( cudaMemcpyToSymbol(c_D, &D_coeff, sizeof(double)) );
    
    checkCuda( cudaMemcpyToSymbol(c_N, &N, sizeof(int)) );
    checkCuda( cudaMemcpyToSymbol(c_M, &M, sizeof(int)) );
    checkCuda( cudaMemcpyToSymbol(c_subDiv, &subDiv, sizeof(double)) );
    checkCuda( cudaMemcpyToSymbol(c_wingL, &wingL, sizeof(int)) );
    checkCuda( cudaMemcpyToSymbol(c_zShift, &zShift, sizeof(double)) );

    double *R = new double[N];
    for(int j=0;j<N;j++) R[j] = RC-RL/2+RL*(double(j)/double(N-1));
    
    double *phi = new double[N*N];
    double *J = new double[N*N]; // Z-flux
    double *K = new double[N*N]; // R-flux
    double *I = new double[N]; 
    for(int i=0; i<N; i++) I[i] = 0.0;

    initPhi(phi,R);

    double *d_R, *d_phi, *d_J, *d_K;
    checkCuda( cudaMalloc((void**)&d_R, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_phi, N*N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_J, N*N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_K, N*N*sizeof(double)) );

    checkCuda( cudaMemcpy(d_R, R, N*sizeof(double), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_phi, phi, N*N*sizeof(double), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemset(d_J, 0, N*N*sizeof(double)) );
    checkCuda( cudaMemset(d_K, 0, N*N*sizeof(double)) );

    const char* kernelInFile = "kernelInput.dat";
    std::cout << "  -> loading kernel... ";
    if(!loadNVecFromDrive(intKernel, kernelInFile, kernelN)){
        fprintf(stderr, "fatal error\n");
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<kernelN; i++) intKernel[i] *= ISF;
    std::cout << "done.\n";
    saveNVecToDrive(intKernel, outDir, "intKernel.dat", kernelN);

    double *d_intKernel;
    checkCuda( cudaMalloc((void**)&d_intKernel, kernelN*sizeof(double)) );
    checkCuda( cudaMemcpy(d_intKernel, intKernel, kernelN*sizeof(double), cudaMemcpyHostToDevice) );  
    checkCuda( cudaMemcpyToSymbol(c_convKernel, intKernel, kernelN*sizeof(double)) );

    double *d_I, *d_psi, *d_percoll, *d_gradWing, *d_alp, *d_b, *d_c, *d_d;
    double *d_psiIntp, *d_IIntp, *d_splineScratch;

    checkCuda( cudaMalloc((void**)&d_splineScratch, 5 * N * sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_I, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_psi, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_percoll, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_gradWing, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_alp, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_b, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_c, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_d, N*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_psiIntp, M*sizeof(double)) );
    checkCuda( cudaMalloc((void**)&d_IIntp, M*sizeof(double)) );

    checkCuda( cudaMemset(d_I, 0, N*sizeof(double)) );
    checkCuda( cudaMemset(d_psi, 0, N*sizeof(double)) );
    checkCuda( cudaMemset(d_percoll, 0, N*sizeof(double)) );
    checkCuda( cudaMemset(d_gradWing, 0, N*sizeof(double)) );
    checkCuda( cudaMemset(d_alp, 0, N*sizeof(double)) );
    checkCuda( cudaMemset(d_psiIntp, 0, M*sizeof(double)) );

    double *gradWing = new double[N];
    for(int i=0; i<N; i++) gradWing[i] = 0.0;
    checkCuda( cudaMemcpy(d_gradWing, gradWing, N*sizeof(double), cudaMemcpyHostToDevice) );
    
    double *percoll = new double[N];
    for(int i=0; i<N; i++) percoll[i] = 0.0;
    checkCuda( cudaMemcpy(d_percoll, percoll, N*sizeof(double), cudaMemcpyHostToDevice) );

    dim3 blockN(256, 1);
    dim3 gridN((N + 255) / 256, 1);
    dim3 block2D(256, 1);
    dim3 grid2D((N + 255) / 256, N);
    dim3 blockM(256, 1);
    dim3 gridM((M + 255) / 256, 1);

    std::cout << "  -> starting simulation loop...\n\n";
    auto sim_start_time = std::chrono::high_resolution_clock::now();
    
    // --- CUDA Event Timing Setup ---
    cudaEvent_t start, stop;
    checkCuda( cudaEventCreate(&start) );
    checkCuda( cudaEventCreate(&stop) );
    
    std::vector<double> step_times;
    step_times.reserve(NT);
    const int SKIP_START = 50;  // Skip warmup
    const int SKIP_END = 50;    // Skip cooldown variance

    int n_out = NO;
    double t = 0.0;

    for (int i = 0; i < NT; i++){
        // Check if this step is an output step
        bool is_output_step = (((i-1) % n_out) == 0) | (i == 1) | (i==NT);
        
        // Determine if we should measure this step
        // We measure if it's NOT an output step, and within our valid window
        bool measure_step = (i > SKIP_START) && (i < (NT - SKIP_END)) && (!is_output_step);

        if(measure_step) {
            cudaEventRecord(start, 0);
        }

        // --- Kernel Execution ---
        CuKernelInte  <<< gridN,  blockN >>> (d_phi, d_psi);
        CuKernelCmpA  <<< gridN,  blockN >>> (d_psi, d_alp);
        CuKernelSplineCoeffs<<<1,1>>>(d_psi, d_alp, d_b, d_c, d_d, d_splineScratch);
        CuKernelSplineEval <<< gridM, blockM >>> (d_psi, d_b, d_c, d_d, d_psiIntp);
        CuKernelConv <<< gridM, blockM, (blockM.x + kernelN - 1) * sizeof(double) >>> (d_psiIntp, d_IIntp, d_intKernel);
        CuKernelDSmp <<< gridN, blockN >>> (d_IIntp, d_I);
        
        CuKernelGrad <<< gridN, blockN >>> (d_percoll, t);
        CuKernelWing <<< gridN, blockN >>> (d_percoll, d_gradWing, t);

        CuKernelComputeFlux <<< grid2D, block2D >>> (d_phi, d_J, d_K, d_percoll, d_R, d_I, d_gradWing);
        CuKernelUpdatePhi   <<< grid2D, block2D >>> (d_phi, d_J, d_K);
        // ------------------------

        if(measure_step) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop); // Wait for this step to finish recording
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            step_times.push_back((double)milliseconds);
        }

        if( is_output_step ){
            // Need to sync before copying memory if we didn't measure/sync above
            if(!measure_step) cudaDeviceSynchronize();

            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - sim_start_time;
            printProgressBar(i, NT, elapsed.count());

            checkCuda( cudaMemcpy(phi, d_phi, N*N*sizeof(double), cudaMemcpyDeviceToHost) );
            double* psi_host = new double[N];
            checkCuda( cudaMemcpy(psi_host, d_psi, N*sizeof(double), cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(gradWing, d_gradWing, N*sizeof(double), cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(percoll, d_percoll, N*sizeof(double), cudaMemcpyDeviceToHost) );

            std::stringstream ss;
            ss << "phi_" << std::setfill('0') << std::setw(10) << i << ".dat";
            saveArrToDrive(phi, outDir, ss.str());

            ss.str(""); ss << "psi_" << std::setfill('0') << std::setw(10) << i << ".dat";
            saveVecToDrive(psi_host, outDir, ss.str());     
            
            ss.str(""); ss << "gw_" << std::setfill('0') << std::setw(10) << i << ".dat";
            saveVecToDrive(gradWing, outDir, ss.str());

            ss.str(""); ss << "gp_" << std::setfill('0') << std::setw(10) << i << ".dat";
            saveVecToDrive(percoll, outDir, ss.str());

            delete[] psi_host;
        }
        t += IT;
    }

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - sim_start_time;
    printProgressBar(NT, NT, elapsed.count());
    
    // --- Statistics Calculation ---
    std::cout << "\n\n  -> simulation complete.\n";

    if(!step_times.empty()) {
        double sum = std::accumulate(step_times.begin(), step_times.end(), 0.0);
        double mean = sum / step_times.size();
        
        double sq_sum = 0.0;
        for(const double &val : step_times) {
            sq_sum += (val - mean) * (val - mean);
        }
        double stdev = std::sqrt(sq_sum / step_times.size());

        std::cout << "  -> avg step time: " 
                  << std::fixed << std::setprecision(4) << mean 
                  << " ms (+/- " << stdev << " ms)\n";
        std::cout << "  -> statistics based on " << step_times.size() 
                  << " pure computation steps (GPU timing)\n";
    } else {
        std::cout << "  -> warning: not enough steps for statistics (increase NT).\n";
    }
    
    checkCuda( cudaEventDestroy(start) );
    checkCuda( cudaEventDestroy(stop) );
    
    checkCuda( cudaFree(d_phi) );
    checkCuda( cudaFree(d_J) );
    checkCuda( cudaFree(d_K) );
    checkCuda( cudaFree(d_R) );
    checkCuda( cudaFree(d_percoll) );
    checkCuda( cudaFree(d_I) );
    checkCuda( cudaFree(d_intKernel) );
    checkCuda( cudaFree(d_psi) );
    checkCuda( cudaFree(d_psiIntp) );
    checkCuda( cudaFree(d_IIntp) );
    checkCuda( cudaFree(d_alp) );
    checkCuda( cudaFree(d_gradWing) );
    checkCuda( cudaFree(d_b) );
    checkCuda( cudaFree(d_c) );
    checkCuda( cudaFree(d_d) );
    checkCuda( cudaFree(d_splineScratch) );

    delete [] phi; delete [] J; delete [] K; delete [] I; delete [] percoll; delete [] gradWing; delete [] R;
}