#include "src/header.h"
#include "src/definitions.h"
#include "src/constants.cu"
#include "src/cuda_kernel.cu"
#include "src/cuda_kernel_sigmoid.cu" // for sigmoidal gradient
//#include "src/cuda_kernel_linear.cu" // for linear gradient
#include "src/functions.cu"

// main function
int main(int argc, char *argv[])
{
    // detect cuda device
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );
    int cudaDevice;
    checkCuda( cudaChooseDevice( &cudaDevice, &prop) );
    printf("\nDevice Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
    // read parameter arguments from command line
    readParameters(argc, argv);
    // run simulation
    runSim();
    return 0;
}