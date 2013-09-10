#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 512
int main(void) {
	int *a, *b, *c;			// Host copies
	int *d_a, *d_b, *d_c;	// Device copies
	int size = N*sizeof(int); // Need integer the size of space

	// Allocate device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	random_ints(a, N);
	random_ints(b, N);

	// Copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on the GPU and pass the parameters
	add<<<N, 1>>>(d_a, d_b, d_c);

	// Copy device results back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	printf("Result: %d", c);

	// Free memory!
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
