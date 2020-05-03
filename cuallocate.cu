#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __host__
void cuAlloc(void **block, size_t numelements, size_t typesize, int commRank)
{
	cudaError_t cE;

	int cudaDeviceCount;

	// Check if enough devices
	if ( (cE = cudaGetDeviceCount( &cudaDeviceCount )) != cudaSuccess ) {
		printf(" Unable to determine cuda device count, error is %d, count is %d\n",
				cE, cudaDeviceCount);
		exit(-1);
	}

	// Assign rank to appropriate device
	if ( (cE = cudaSetDevice( commRank % cudaDeviceCount )) != cudaSuccess) {
		printf(" Unable to have rank %d set to cuda %d, error is %d \n",
				commRank, (commRank % cudaDeviceCount), cE);
		exit(-1);
	}

	cudaMallocManaged(block, numelements*typesize);
}

extern "C" __host__
void cuAllocDouble(double **block, size_t numelements, int commRank)
{
	size_t i;
	cudaError_t cE;

	int cudaDeviceCount;

	// Check if enough devices
	if ( (cE = cudaGetDeviceCount( &cudaDeviceCount )) != cudaSuccess ) {
		printf(" Unable to determine cuda device count, error is %d, count is %d\n",
				cE, cudaDeviceCount);
		exit(-1);
	}

	// Assign rank to appropriate device
	if( (cE = cudaSetDevice( commRank % cudaDeviceCount )) != cudaSuccess) {
		printf(" Unable to have rank %d set to cuda %d, error is %d \n",
				commRank, (commRank % cudaDeviceCount), cE);
		exit(-1);
	}

	cudaMallocManaged(block, numelements*sizeof(double));
	for (i=0; i<numelements; i++) (*block)[i] = 0.0;

}

extern "C" __host__
void cuFreeDouble(double *block)
{
	cudaFree(block);
}

extern "C" __host__
void cuFree(void *block)
{
	cudaFree(block);
}
