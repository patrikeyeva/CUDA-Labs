
#include <fstream>
#include "stdlib.h"
#include <stdio.h>


#define CSC(call)												\
do {															\
	cudaError_t res = call;										\
	if (res != cudaSuccess) {									\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",	    \
				__FILE__, __LINE__, cudaGetErrorString(res));	\
		exit(0);												\
	}															\
} while (0)



__global__ void kernel(char* a, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //номер текущей нити 
	printf("%s (current thread number %d)\n", a, idx);

}



int main() {
	const int n = 14;
	char chars[n] = "Hello, World!";

	char *arr_dev;
	CSC(cudaMalloc(&arr_dev, sizeof(char) * n));
	CSC(cudaMemcpy(arr_dev, chars, sizeof(char) * n, cudaMemcpyHostToDevice));
	CSC(cudaGetLastError());

	kernel << < 2, 4 >> > (arr_dev, n); 
	cudaDeviceSynchronize();
	CSC(cudaGetLastError());

	CSC(cudaFree(arr_dev));

	return 0;

}
