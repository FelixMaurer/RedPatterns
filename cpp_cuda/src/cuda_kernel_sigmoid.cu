/* sigmoid parameters */
#define b2 3.1773e-4
#define b3 (gradL/2)
#define b4 0.0338
#define b5 1.1012e-3
#define b6 0.6
#define b7 1.5205
/* sigmoidal percoll gradient kernel */
__global__ void CuKernelGrad(double* percoll, double t)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    double x = c_IZ * double(i);
    if(i>double(c_N-1)/2)
        percoll[i] =  + b2*pow(t,b7)*(x-c_zShift-b3)/b4/pow(1-pow((x-c_zShift-b3)/b4,b5*t+b6),1/(b5*t+b6));
    if(i<double(c_N-1)/2)
        percoll[i] =  - b2*pow(t,b7)*(-x+c_zShift+b3)/b4/pow(1-pow((-x+c_zShift+b3)/b4,b5*t+b6),1/(b5*t+b6));
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
    r1 = r2-50;
    x1 = 12;
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
}