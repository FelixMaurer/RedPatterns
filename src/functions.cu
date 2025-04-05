/* saving 2D arrays to disk */
void saveArrToDrive(double* f, char* outFileName)
{
    const uint16_t sampleSkip = ceil(double(N)/256.0f);
    std::ofstream ofs(outFileName);
    for(int i=0;i<N;i+=sampleSkip){
        for(int j=0;j<N;j+=sampleSkip){
            if(j>0)
                ofs << "\t";
            ofs << f[i+N*j]; 
        }
        ofs << "\n";
    }
    ofs.close();
}
/* saving interpolation to disk */
void saveIntVecToDrive(double* f, char* outFileName)
{
    const uint16_t sampleSkip = 1;
    std::ofstream ofs(outFileName);
    for(int i=0;i<M;i+=sampleSkip){
        if(i>0)
            ofs << "\t";
        ofs << f[i]; 
    }
    ofs << "\n";
    ofs.close();
}
/* saving vector to disk */
void saveVecToDrive(double* f, char* outFileName)
{
    const uint16_t sampleSkip = ceil(double(N)/256.0f);
    std::ofstream ofs(outFileName);
    for(int i=0;i<N;i+=sampleSkip){
        if(i>0)
            ofs << "\t";
        ofs << f[i]; 
    }
    ofs << "\n";
    ofs.close();
}
/* saving n-vector to disk */
void saveNVecToDrive(double* f, char* outFileName, int n)
{
    std::ofstream ofs(outFileName);
    for(int i=0;i<n;i+=1){
        if(i>0)
            ofs << "\t";
        ofs << f[i]; 
    }
    ofs << "\n";
    ofs.close();
}
/* kernel function */
#define fLJ(r,sigma) (4*U*(12*pow(sigma,12)/pow(r,13)-6*pow(sigma,6)/pow(r,7)))
#define g(r,d,sigmaC) (8.41e4*exp(-pow(r-d,2)/(2*pow(sigmaC,2))))
void genConvKernel(){
    // compute effective potential 
    double kernelL = (double(kernelN)-1)*IZ/subDiv;
    double kernelDZ = kernelL/double(kernelN-1);
    double subRes = 10000;
    double fineRes = subRes*(double(kernelN+1)/2);
    double force;
    double fineR;
    double gpdf;
    double kernelFine[int(fineRes)];
    double fineDR = kernelDZ/subRes; // only take positive values
    double sigma = 5.6e-6;
    double sigmaC = 0.5e-6;
    double eqDist = 6.58546720106423709125472581993321341542468871921300888061523437500000000000000000e-06;
    // use central interval positions
    double sum = 0;
    kernelFine[0] = 0; // avoid divergence of force term at zero
    for(int i=1;i<fineRes;i++){
        fineR = double(i*fineDR);
        force = fLJ(fineR,sigma);
        gpdf = g(fineR,eqDist,sigmaC);
        if(fineR<1e-8) // make up for numerical error near divergence
            gpdf = 0.0;
        kernelFine[i] = sum; // compute integral
        sum = sum + fineDR*force*gpdf;
    }
    
    // integration constant
    for(int i=0;i<fineRes;i++)
        kernelFine[i] = kernelFine[int(fineRes)-1]-kernelFine[i];
    // sampling of kernel
    intKernel[(kernelN+1)/2] = 0;
    double kernelZ;
    for(int i=(kernelN+1)/2;i<kernelN;i++){
        kernelZ = double(i*kernelDZ)-kernelL/2;
        intKernel[i] = kernelZ*kernelFine[int((i+1-double(kernelN+1)/2)*subRes)];
        intKernel[kernelN-1-i] = -intKernel[i];    
    }
    printf("kernel length = %.32e m\n",kernelL);
}
/* check cuda device */
inline
cudaError_t checkCuda(cudaError_t result)
{
    #if defined(DEBUG) || defined(_DEBUG)
        if (result != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            assert(result == cudaSuccess);
        }
    #endif
    return result;
}
/* initial values for phi */
void initPhi(double *f, double *R)
{
    double edgeZ = wingL+2;
    double edgeR = wingL;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            f[i+N*j] = exp(-pow(R[j]-(Rmu),2)/(2.0*pow(Rsigma,2)));
            if((i<edgeZ)|(i>(N-1-edgeZ)))
                f[i+N*j] = 0.0;
            if((j<edgeR)|(j>(N-1-edgeR)))
                f[i+N*j] = 0.0;
        }
    // normalization
    /*
    integral phi dz drho = intgral psi dz = L N <psi> = L N PSI
    sum phi IZ = N <psi> = N PSI
    */
    double phiSum = 0.0;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            phiSum += f[i+N*j];
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            f[i+N*j] = f[i+N*j]/phiSum*PSI*(N-2*edgeZ);
}
/* taking arguments */
void readParameters(int argc, char *argv[]){
    int argIdx = 1;
    // U
    if(argc>argIdx)
        U = std::stod(argv[argIdx]);
    argIdx++;
    // PSI
    if(argc>argIdx)
        PSI = std::stod(argv[argIdx]);
    argIdx++;
    // IT
    if(argc>argIdx)
        IT = std::stod(argv[argIdx]);
    argIdx++;
    // T
    if(argc>argIdx)
        T = std::stod(argv[argIdx]);
    argIdx++;
    // NO
    if(argc>argIdx)
        NO = std::stod(argv[argIdx]);
    argIdx++;
    // gamma
    if(argc>argIdx)
        h_gamma = std::stod(argv[argIdx]);
    argIdx++;
    // delta
    if(argc>argIdx)
        h_delta = std::stod(argv[argIdx]);
    argIdx++;
    // kappa
    if(argc>argIdx)
        h_kappa = std::stod(argv[argIdx]);
    argIdx++;
    // re-evalutate parameters
    NT = ceil(T/IT);
}
/* running simulation */
void runSim(){
    // allocate space for output filename
    char outFileName[19];
    // constants to device memory
    checkCuda( cudaMemcpyToSymbol(c_IZ, &IZ, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_IT, &IT, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_PSI, &PSI, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_beta, &h_beta, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_alpha, &h_alpha, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_gamma, &h_gamma, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_delta, &h_delta, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_kappa, &h_kappa, sizeof(double), 0, cudaMemcpyHostToDevice) );
    // coordinates
    printf("writing coordinate arrays to GPU mem.\n");
    double *R = new double[N]; // density dimension vector
    for(int j=0;j<N;j++)
        R[j] = RC-RL/2+RL*(double(j)/double(N-1));
    int bytes = 0; // size of array or vector
    // R device array
    bytes = N*sizeof(double);
    double* d_R; // R on device
    // (1) allocate
    checkCuda( cudaMalloc((void**)&d_R, bytes) );
    // (2) write initial values
    checkCuda( cudaMemcpy(d_R, R, bytes, cudaMemcpyHostToDevice) );  
    // arrays of volumetric density and flux
    printf("writing flux and density arrays to GPU mem.\n");
    double *phi = new double[N*N];
    double *dJ = new double[N*N];
    double *J = new double[N*N];
    // write initial values (calculated from R)
    initPhi(phi,R);
    /* write initial condition to drive
    sprintf(outFileName,"initPhi.dat");
    saveArrToDrive(phi,outFileName);*/
    // device arrays
    bytes = N*N*sizeof(double);
    double *d_phi, *d_dJ, *d_J;
    // (1) allocate
    checkCuda( cudaMalloc((void**)&d_phi, bytes) );
    checkCuda( cudaMalloc((void**)&d_dJ, bytes) );
    checkCuda( cudaMalloc((void**)&d_J, bytes) );
    // (2) write initial values
    checkCuda( cudaMemcpy(d_phi, phi, bytes, cudaMemcpyHostToDevice) );  
    checkCuda( cudaMemset(d_dJ, 0, bytes) );
    checkCuda( cudaMemset(d_J, 0, bytes) );
    printf("writing vectors to GPU mem.\n");

    /* interaction kernel */
    genConvKernel();
    sprintf(outFileName,"intKernel.dat");
    saveNVecToDrive(intKernel,outFileName,kernelN);
    bytes = kernelN*sizeof(double);
    double *d_intKernel;
    // (1) allocate
    printf("allocate intkernel.\n");
    checkCuda( cudaMalloc((void**)&d_intKernel, bytes) );
    // (2) write initial values
    printf("write intkernel.\n");
    checkCuda( cudaMemcpy(d_intKernel, intKernel, bytes, cudaMemcpyHostToDevice) );  

    /* interaction integral */
    double* psi = new double[N]; // for gathering data from device
    double* I = new double[N]; // for gathering data from device
    bytes = N*sizeof(double);
    double *d_I;
    // (1) allocate
    printf("allocate integral.\n");
    checkCuda( cudaMalloc((void**)&d_I, bytes) );
    // (2) write initial values
    printf("write integral.\n");
    checkCuda( cudaMemset(d_I, 0, bytes) );

    /* psi - volume fraction */
    printf("allocate psi.\n");
    bytes = N*sizeof(double);
    double *d_psi;
    // (1) allocate
    checkCuda( cudaMalloc((void**)&d_psi, bytes) );
    printf("write psi.\n");
    // (2) write initial values
    checkCuda( cudaMemset(d_psi, 0, bytes) );

    /* interpolated psi */
    printf("allocate interpolated psi.\n");
    bytes = sizeof(double)*M;
    double *d_psiIntp;
    // (1) allocate
    checkCuda( cudaMalloc((void**)&d_psiIntp, bytes) );
    printf("write psi.\n");
    // (2) write initial values
    checkCuda( cudaMemset(d_psiIntp, 0, bytes) );

    /* interpolated I integral */
    printf("allocate interpolated I.\n");
    bytes = sizeof(double)*M;
    double *d_IIntp;
    // (1) allocate
    checkCuda( cudaMalloc((void**)&d_IIntp, bytes) );
    printf("write psi.\n");
    // (2) write initial values
    checkCuda( cudaMemset(d_IIntp, 0, bytes) );

    /* percoll - gradient */
    printf("allocate percoll.\n");
    double *percoll = new double[N];
    for(int k=0; k<N; k++)
        percoll[k] = 0.0;

    // percoll device array
    bytes = N*sizeof(double);
    double* d_percoll; // R on device
    // (1) allocate
    checkCuda( cudaMalloc((void**)&d_percoll, bytes) );
    // (2) write initial values
    checkCuda( cudaMemcpy(d_percoll, percoll, bytes, cudaMemcpyHostToDevice) );
    
    // gradient wing
    printf("allocate gradient wing.\n");
    double *gradWing = new double[N];
    for(int i=0; i<N; i++)
        gradWing[i] = 0.0;
    // gradient wing device array
    bytes = N*sizeof(double);
    double* d_gradWing; // R on device
    // (1) allocate
    checkCuda( cudaMalloc((void**)&d_gradWing, bytes) );
    // (2) write initial values
    checkCuda( cudaMemcpy(d_gradWing, gradWing, bytes, cudaMemcpyHostToDevice) );
    
    // arrays for interpolation computation
    bytes = (M-1)*sizeof(double);
    double * d_alp;
    checkCuda( cudaMalloc((void**)&d_alp, bytes) );
    checkCuda( cudaMemset(d_alp, 0, bytes) );
    
    // output interpolation
    double psiIntp[int(M)];
    
    printf("starting timer.\n");
    // start time measurement
    float milliseconds;
    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    printf("defining grid and starting loop.\n");
    // Kernel invocation
    int nBlocksX, nBlocksY, nThreadsX, nThreadsY;
    // grid layout, usually max threads in X dimension (1024)
 
    nThreadsX = N;
    nThreadsY = 1;
    nBlocksX = 1;
    nBlocksY = N;

    dim3 numBlocks(nBlocksX,nBlocksY);
    dim3 threadsPerBlock(nThreadsX,nThreadsY);

    dim3 numBlocksA(subDiv,1);
    dim3 threadsPerBlockA(N,1);

    dim3 numBlocksD(1,1);
    dim3 threadsPerBlockD(N,1);

    printf("N = %d, M = %d\n",N,M);
    printf("alpha = %.32e\nbeta = %.32e\n",h_alpha,h_beta);
    printf("gamma = %.32e\ndelta = %.32e\nkappa = %.32e\n",h_gamma,h_delta,h_kappa);
    printf("system size L = %.32e m\n",sysL);
    printf("increment size dz = %.32e m\n",IZ);
    printf("launching with\n nBlocksX\t| nThreadsX\t| nBlocksY\t| nThreadsY\n %d\t\t| %d\t\t| %d\t\t| %d\n",nBlocksX,nThreadsX,nBlocksY,nThreadsY);
    checkCuda( cudaEventRecord(startEvent, 0) );
    // iteration loop
    int n_out = NO;
    double t = 0.0;
    for (int i = 0; i < NT; i++){
        /* integration */
        CuKernelInte <<< numBlocks, threadsPerBlock >>> (d_phi,d_psi);
        /* interpolation */
        CuKernelCmpA <<< numBlocksA, threadsPerBlockA >>> (d_psi, d_alp);
        CuKernelCmpL <<< numBlocksA, threadsPerBlockA >>> (d_psi, d_alp, d_psiIntp);
        CuKernelConv <<< numBlocksA, threadsPerBlockA >>> (d_psiIntp,d_IIntp,d_intKernel);
        CuKernelDSmp <<< numBlocksD, threadsPerBlockD >>> (d_IIntp, d_I);
        /* density gradient */
        CuKernelGrad <<< numBlocks, threadsPerBlock >>> (d_percoll, t);
        CuKernelWing <<< numBlocks, threadsPerBlock >>> (d_percoll, d_gradWing, t);
        /* iteration */
        CuKernelIter <<< numBlocks, threadsPerBlock >>> (d_phi, d_J, d_dJ, d_percoll, d_R, d_I,d_psi,d_intKernel,t,d_gradWing);
        if( (((i-1) % n_out) == 0) | (i == 1) | (i==NT)){
            // retrieve data from GPU mem
            bytes = N*N*sizeof(double);
            checkCuda( cudaMemcpy(phi, d_phi, bytes, cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(J, d_J, bytes, cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(dJ, d_dJ, bytes, cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(I, d_I, N*sizeof(double), cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(psi, d_psi, N*sizeof(double), cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(psiIntp, d_psiIntp, N*sizeof(double)*subDiv, cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(gradWing, d_gradWing, N*sizeof(double), cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(percoll, d_percoll, N*sizeof(double), cudaMemcpyDeviceToHost) );
            //checkCuda( cudaMemcpy(IIntp, d_IIntp, N*sizeof(double)*subDiv, cudaMemcpyDeviceToHost) );
            // write data to file
            sprintf(outFileName,"phi_%010d.dat",i);
            saveArrToDrive(phi,outFileName);

            sprintf(outFileName,"psi_%010d.dat",i);
            saveVecToDrive(psi,outFileName);     
            
            sprintf(outFileName,"gW_%010d.dat",i);
            saveVecToDrive(gradWing,outFileName);

            sprintf(outFileName,"gP_%010d.dat",i);
            saveVecToDrive(percoll,outFileName);

            /* optional output
            sprintf(outFileName,"J_%010d.dat",i);
            saveArrToDrive(J,outFileName);
            sprintf(outFileName,"dJ_%010d.dat",i);
            saveArrToDrive(dJ,outFileName);
            sprintf(outFileName,"I_%010d.dat",i);
            saveVecToDrive(I,outFileName);
            

            sprintf(outFileName,"pit_%010d.dat",i);
            saveIntVecToDrive(psiIntp,outFileName);
            */
           
            // measure time
            checkCuda( cudaEventRecord(stopEvent, 0) );
            checkCuda( cudaEventSynchronize(stopEvent) );
            checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );
            printf("step: %d/%d\n", i, NT);
            printf("runtime (sec): %.5f\n", milliseconds/1000.0);
            printf("remaining (sec): %.5f\n", milliseconds/1000.0 * (NT-i)/i);
       }
       t += IT;
    }
    printf("finished.\n\n");
    // stop timer
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

    // show stats    
    printf("   total steps: %d\n", NT);
    printf("   total time (ms): %f\n", milliseconds);
    printf("   average time (ms): %f\n", milliseconds / NT);

    // delete arrays and free memory
    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );

    checkCuda( cudaFree(d_phi) );
    checkCuda( cudaFree(d_dJ) );
    checkCuda( cudaFree(d_J) );
    checkCuda( cudaFree(d_R) );
    checkCuda( cudaFree(d_percoll) );
    checkCuda( cudaFree(d_I) );
    checkCuda( cudaFree(d_intKernel) );
    checkCuda( cudaFree(d_psi ) );
    checkCuda( cudaFree(d_psiIntp ) );
    checkCuda( cudaFree(d_IIntp ) );
    checkCuda( cudaFree(d_alp ) );
    checkCuda( cudaFree(d_gradWing ) );

    delete [] phi;
    delete [] dJ;
    delete [] J;
    delete [] I;
}