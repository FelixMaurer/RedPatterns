/* linear percoll gradient kernel */
#define PL 8.0
__global__ void CuKernelGrad(double* percoll, double t)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    double x = c_IZ * double(i);
    percoll[i] =  (x-c_zShift-gradL/2)/(gradL/2)*PL/2;
}
__global__ void CuKernelWing(double* percoll, double* gradWing, double t)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // compute gradient wing
    double r1,r2,r3;
    double x1,x2;
    r3 = (percoll[c_wingL]-percoll[c_wingL-1]);
    r2 = percoll[c_wingL];
    r1 = r2-10;
    x1 = 20;
    x2 = c_wingL;                                                 
    if(percoll[int(x1)] < r1)
        r1 = percoll[int(x1)];

    double a,b,c; // parameters of parabola
    a = (r1-r2+r3*(x2-x1))/((x1-x2)*(x1-x2));
    b = r3-2*a*x2;
    c = r2-r3*x2+x2*x2*a;
    gradWing[i] = 0.0;
    if(i<=c_wingL)
        gradWing[i] = a*i*i + b * i + c;
    if(i>=c_N-1-c_wingL)
        gradWing[i] = -(a*(c_N-1-i)*(c_N-1-i) + b * (c_N-1-i) + c);
    if(i>=c_N-1-13)
        gradWing[i] = gradWing[c_N-1-13];
    if(i<=13)
        gradWing[i] = gradWing[13];  
}