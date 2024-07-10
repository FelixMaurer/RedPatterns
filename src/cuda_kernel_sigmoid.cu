/* Sigmoidal Percoll gradient kernel */
__global__ void CuKernelGrad(double* percoll, double t)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    double x = IX * double(i);
    if(i>double(N-1)/2)
        percoll[i] =  + b2*pow(t,b7)*(x-xShift-b3)/b4/pow(1-pow((x-xShift-b3)/b4,b5*t+b6),1/(b5*t+b6));
    if(i<double(N-1)/2)
        percoll[i] =  - b2*pow(t,b7)*(-x+xShift+b3)/b4/pow(1-pow((-x+xShift+b3)/b4,b5*t+b6),1/(b5*t+b6));
}
__global__ void CuKernelWing(double* percoll, double* gradWing, double t)
{
    // get indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // compute gradient wing
    double r1,r2,r3;
    double x1,x2;
    r3 = (percoll[wingL]-percoll[wingL-1]);
    r2 = percoll[wingL];
    r1 = r2-50;
    x1 = 7;
    x2 = wingL;                                                 
    if(percoll[int(x1)] < r1)
        r1 = percoll[int(x1)];

    double a,b,c; // parameters of parabola
    a = (r1-r2+r3*(x2-x1))/((x1-x2)*(x1-x2));
    b = r3-2*a*x2;
    c = r2-r3*x2+x2*x2*a;

    gradWing[i] = 0.0;
    if(i<=wingL)
        gradWing[i] = a*i*i + b * i + c;
    if(i>=N-1-wingL)
        gradWing[i] = -(a*(N-1-i)*(N-1-i) + b * (N-1-i) + c);
}