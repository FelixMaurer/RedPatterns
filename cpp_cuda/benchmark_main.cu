#include "src/header.h"
#include "src/definitions.h"

// --- Forward Declarations ---
// These allow main() to call functions defined in the included .cu files below
cudaError_t checkCuda(cudaError_t result);
void readParameters(int argc, char *argv[]);
void runSim();

// --- Unity Build Includes ---
// Including .cu files directly means we compile ONLY main.cu
#include "src/constants.cu"
#include "src/cuda_kernel.cu"
// Select ONE gradient kernel:
#include "src/cuda_kernel_linear.cu" 
// #include "cuda_kernel_sigmoid.cu" 
#include "src/functions.cu"

int main(int argc, char *argv[])
{
    // detect cuda device
    cudaDeviceProp prop;
    // We can't use checkCuda here easily if it's not defined yet, 
    // but the forward declaration above fixes that.
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