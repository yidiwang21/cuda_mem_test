#include <stdint.h>

__device__ inline uint64_t GlobalTimer64(void) {
    volatile uint64_t reading;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(reading));
    return reading;
}

static __device__ __inline__ unsigned int GetSMID(void) {
    unsigned int ret;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}




static __global__ void kernel_func(uint64_t in) {
    uint64_t curr_time = 0;
    for (uint64_t i = 0; i < in; i++) {
        // may conflict when reading from the same reg concurrently
        // same results were gotten when I bypass this
        curr_time = GlobalTimer64();
     /* 
        int temp = temp % i;
        for (uint64_t j = 0; j < 2; j++) {
            temp %= j;
            temp += threadIdx.x;
        } */

    }
}