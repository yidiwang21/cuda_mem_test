#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thread>
#include <sys/wait.h> 
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include "src/kernel.cu"
#include "src/support.cu"

using namespace std;

// for nvidia xavier
#define SM_NUM  8

int main (int argc, char *argv[]) {
    remove( "log_tranditional.txt" );
    std::string outfile = "log_tranditional.txt";

    cudaError_t cuda_ret;


    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize = 10000000;

    int cp = fork();

    if (cp > 0) {
        A_sz = VecSize;
        B_sz = VecSize;
        C_sz = VecSize;
        A_h = (float*) malloc( sizeof(float)*A_sz );
        for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }
    
        B_h = (float*) malloc( sizeof(float)*B_sz );
        for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }
    
        C_h = (float*) malloc( sizeof(float)*C_sz );
        Timer timer;
    
        printf("Size Of vector: %u x %u\n  ", VecSize);
    
        // Allocate device variables ----------------------------------------------
    
        printf("Allocating device variables..."); fflush(stdout);
        startTime(&timer);
    
        //INSERT CODE HERE
        size_t bytes = sizeof(float) * VecSize;
        cudaMalloc((void**) &A_d, bytes);
        cudaMalloc((void**) &B_d, bytes);
        cudaMalloc((void**) &C_d, bytes);
    
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        std::string end_time0 = std::to_string(elapsedTime(timer) * 1000);
    
        // Copy host variables to device ------------------------------------------
    
        printf("Copying data from host to device..."); fflush(stdout);
    
        //INSERT CODE HERE
        cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);
    
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        std::string end_time1 = std::to_string(elapsedTime(timer) * 1000);
    
        // Launch kernel  ---------------------------
        basicVecAdd(A_d, B_d, C_d, VecSize); //In kernel.cu

        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) {
            // FATAL("Unable to launch kernel");
            fprintf(stderr, "Unable to launch kernel\n");
            exit(-1);
        }
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        std::string end_time2 = std::to_string(elapsedTime(timer) * 1000);

        // Copy device variables from host ----------------------------------------
    
        printf("Copying data from device to host..."); fflush(stdout);
    
        //INSERT CODE HERE
        cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);
    
        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        std::string end_time3 = std::to_string(elapsedTime(timer) * 1000);
    
    
    
        // Free memory ------------------------------------------------------------
    
        free(A_h);
        free(B_h);
        free(C_h);
    
        //INSERT CODE HERE
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);

        
        usleep(elapsedTime(timer) * 1000 * 1000 * 1);
        system("/home/nvidia/tegrastats --stop");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        std::string stop_time = std::to_string(elapsedTime(timer) * 1000);

        std::ofstream out;
        out.open(outfile, std::ios::app);
        out << "time0: " << end_time0 << std::endl;     // after alloc mem
        out << "time1: " << end_time1 << std::endl;     // after copied to device from host
        out << "time2: " << end_time2 << std::endl;     // after kernel execution
        out << "time3: " << end_time2 << std::endl;     // after copied to host from device
        out << "stoptime: " << stop_time << std::endl;  // after cooled down

        wait(NULL);
    }else if (cp == 0) {
        system(("/home/nvidia/tegrastats --interval 10 --logfile " + outfile).c_str());
    }


    return 0;

}