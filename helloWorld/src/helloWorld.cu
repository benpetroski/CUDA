#include <stdio.h>
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello(){
	printf("Hello World! I'm a thread in block %d\n", blockIdx.x);
}

int main(int argc, char **argv){
	//launch kernel
	hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
	//force the prints to flush
	cudaDeviceSynchronize();

	return 0;
}
