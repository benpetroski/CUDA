#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N (3*10)

//void saxpy_cpu(int n, float a, float *x, float *y){
//	for(int i=0; i<n; ++i)
//		y[i]=a*x[i]+y[i];
//}

__global__ void saxpy_gpu(int n, float a, float *x, float *y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n)
		y[i] = a*x[i] + y[i];
}

int main(void){
	float *x, *y; //host copies
	float *d_x, *d_y; //device copies
	int size = N*sizeof(float);

	//Allocate space for device copies
	cudaMalloc((void **)&d_x, size);
	cudaMalloc((void **)&d_y, size);

	//Allocate space for host copies of x and y and setup input values
	x = (float *)malloc(size);
	random_floats(x, N);
	y = (float *)malloc(size);
	random_floats(y, N);

	//Copy input to device
	cudaMemcpy(d_x, &x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, &y, size, cudaMemcpyHostToDevice);

	saxpy_gpu<<<3, 10>>>(N, 2.0f, d_x, d_y);
	cudaDeviceSynchronize();

	//Copy result back to host
	cudaMemcpy(d_y, &y, size, cudaMemcpyDeviceToHost);

	//Cleanup
	cudaFree(d_x); cudaFree(d_y);
	free(x); free(y);
	return 0;
}
