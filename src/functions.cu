/* saving 2D arrays to disk */
void array_to_drive(double* f, char* outFileName)
{
    const uint16_t sampleSkip = ceil(double(N)/512.0f);
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
void interpolation_to_drive(double* f, char* outFileName)
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
void vector_to_drive(double* f, char* outFileName)
{
    const uint16_t sampleSkip = ceil(double(N)/512.0f);
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
void n_vec_to_drive(double* f, char* outFileName, int n)
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
void intKernelFunc(){
    double kernelL = (double(kernelN)-1)*IX/subDiv;
    double kernelX[kernelN];
    for(int i=0;i<kernelN;i++)
        kernelX[i] = -kernelL/2 + double(i)/(kernelN-1) * kernelL; 
    for(int i=0;i<kernelN;i++){
        intKernel[i] = kernelX[i]*( epsilonRep*exp(-pow(kernelX[i],nRep)/(pow(sigmaRep,nRep))) - epsilonAtt*exp(-pow(kernelX[i],nAtt)/(pow(sigmaAtt,nAtt))) );
    }
    printf("kernel size kL = %.32e m\n",kernelL);
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
    double edge = wingL-5;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            f[i+N*j] = exp(-pow(R[j]-(Rmu),2)/(2.0*pow(Rsigma,2)));
            if((i<edge)|(i>(N-1-edge)))
                f[i+N*j] = 0.0;
            if((j<edge)|(j>(N-1-edge)))
                f[i+N*j] = 0.0;
        }
    // normalization
    /*
    integral phi dx dr = intgral psi dx = L N <psi> = L N PSI
    sum phi IX = N <psi> = N PSI
    */
    double phiSum = 0.0;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            phiSum += f[i+N*j];
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            f[i+N*j] = f[i+N*j]/phiSum*PSI*(N-2*edge);
}
/* running simulation */
void runSim(){
    // allocate space for output filename
    char outFileName[19];
    // constants to device memory
    checkCuda( cudaMemcpyToSymbol(c_IX, &IX, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_beta, &h_beta, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpyToSymbol(c_alpha, &h_alpha, sizeof(double), 0, cudaMemcpyHostToDevice) );
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
    array_to_drive(phi,outFileName);*/
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
    intKernelFunc();
    sprintf(outFileName,"intKernel.dat");
    n_vec_to_drive(intKernel,outFileName,kernelN);
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
    double *x = new double[N];
    for(int k=0; k<N; k++)
        x[k] = IX * (k+0.0);
    for(int k=0; k<N; k++)
        percoll[k] = IX * (k+0.0);

    /* write percoll gradient to drive
    sprintf(outFileName,"percoll.dat");
    vector_to_drive(percoll,outFileName);*/
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
    for(int i=1; i<N; i++)
        gradWing[i] = 0.0;
    /* write gradient wing to drive
    sprintf(outFileName,"gradWing.dat");
    vector_to_drive(gradWing,outFileName);*/
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
    printf("eA = %.80e \n",epsilonAtt);
    printf("eR = %.80e \n",epsilonRep);
    printf("system size L = %.32e m\n",XL);
    printf("increment size dx = %.32e m\n",IX);
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
            //checkCuda( cudaMemcpy(IIntp, d_IIntp, N*sizeof(double)*subDiv, cudaMemcpyDeviceToHost) );
            // write data to file
            sprintf(outFileName,"phi_%010d.dat",i);
            array_to_drive(phi,outFileName);

            sprintf(outFileName,"psi_%010d.dat",i);
            vector_to_drive(psi,outFileName);            
            /* optional output
            sprintf(outFileName,"J_%010d.dat",i);
            array_to_drive(J,outFileName);
            sprintf(outFileName,"dJ_%010d.dat",i);
            array_to_drive(dJ,outFileName);
            sprintf(outFileName,"I_%010d.dat",i);
            vector_to_drive(I,outFileName);
            sprintf(outFileName,"gW_%010d.dat",i);
            vector_to_drive(gradWing,outFileName);

            sprintf(outFileName,"pit_%010d.dat",i);
            interpolation_to_drive(psiIntp,outFileName);
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