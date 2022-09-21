
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace std;

#define CSC(call)												\
do {															\
	cudaError_t res = call;										\
	if (res != cudaSuccess) {									\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",	    \
				__FILE__, __LINE__, cudaGetErrorString(res));	\
		exit(0);												\
	}															\
} while (0)



__global__ void change(double *mat, int n, int ind, int max_r) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;  // Абсолютный номер потока
	int offsetx = gridDim.x * blockDim.x;  // Общее кол-во потоков


	for (int i = idx + ind; i < 2 * n; i += offsetx)
	{
		double c = mat[ind + n * i];
		mat[ind + n * i] = mat[n * i + max_r];
		mat[n * i + max_r] = c;
	}

}

__global__ void gauss1(double *mat, int n, int ind) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;  //// Абсолютный номер потока
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;  // Общее кол-во потоков
	int offsety = blockDim.y * gridDim.y;


	for (int i = idx + ind + 1; i < n; i += offsetx) {
		for (int j = idy + ind + 1; j < 2 * n; j += offsety) {
			mat[j * n + i] += mat[j * n + ind] * (-mat[ind * n + i] / mat[ind * n + ind]);
		}

	}

}

__global__ void gauss2(double *mat, int n, int ind) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;  //// Абсолютный номер потока
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;  // Общее кол-во потоков
	int offsety = blockDim.y * gridDim.y;

	for (int i = ind - 1 - idx; i >= 0; i -= offsetx) {
		for (int j = idy + n; j < 2 * n; j += offsety) {
			mat[j * n + i] += mat[j * n + ind] * (-mat[ind * n + i] / mat[ind * n + ind]);
		}
	}
}


struct comparator {
	__host__ __device__ bool operator()(double a, double b) {		// Функция которая сравнивает объекты на "<"
		return abs(a) < abs(b); 									// operator() - переопределение оператора "()" для экземпляра этой структуры
	}
};

int main() {

	int n; //размер матрицы
	scanf("%d", &n);
	double *matrix = (double*)malloc(sizeof(double) * 2 * n * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			scanf("%lf", &matrix[j * n + i]);
		}
	}
	printf("\n");

	// добавляем единичную
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) {
				matrix[i * n + j + n * n] = 1.0;
			}
			else {
				matrix[i * n + j + n * n] = 0.0;
			}
		}
	}


	double *dev_matrix;
	cudaMalloc(&dev_matrix, sizeof(double) * 2 * n * n);
	cudaMemcpy(dev_matrix, matrix, sizeof(double) * 2 * n * n, cudaMemcpyHostToDevice);

	comparator comp;
	thrust::device_ptr<double> ptr = thrust::device_pointer_cast(dev_matrix);

	for (int i = 0; i < n - 1; i++) {

		thrust::device_ptr<double> p_max = thrust::max_element(ptr + i + n * i, ptr + n * (i + 1), comp);
		int max_row = p_max - ptr - n * i;

		if (max_row != i)
		{
			change << <256, 256 >> > (dev_matrix, n, i, max_row);
			CSC(cudaGetLastError());
		}
		gauss1 << < dim3(32, 16), dim3(32, 16) >> > (dev_matrix, n, i);
		CSC(cudaGetLastError());
	}

	for (int i = n - 1; i > 0; i--) {
		gauss2 << < dim3(32, 16), dim3(32, 16) >> > (dev_matrix, n, i);
		CSC(cudaGetLastError());
	}

	cudaMemcpy(matrix, dev_matrix, sizeof(double) * 2 * n * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_matrix);
	CSC(cudaGetLastError());


	for (int i = 0; i < n; i++) {
		for (int j = n; j < 2 * n; j++) {
			matrix[j * n + i] /= matrix[i * n + i];
		}
	}


	for (int i = 0; i < n; ++i) {
		for (int j = n; j < 2 * n; ++j) {
			printf("%.5lf ", matrix[j * n + i]);

		}
		printf("\n");
	}

	free(matrix);
	return 0;
}