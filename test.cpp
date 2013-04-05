#include <iostream>
#include <cmath>
#include <omp.h>
#include <xmmintrin.h>
#define TIMING
#ifdef TIMING
#include <sys/time.h>
#endif


int main() {
#ifdef TIMING
    struct timeval start_time;
    struct timeval stop_time;
    long long total_time = 0;
    __m128 HXRES = _mm_set1_ps(700.0f);

#endif

#ifdef TIMING
    /* record starting time */
    gettimeofday(&start_time, NULL);
#endif

    __m128 cx4, cy4;
    cx4 = _mm_set1_ps((float) 60);

#ifdef TIMING        
    gettimeofday(&stop_time, NULL);
    total_time += (stop_time.tv_sec - start_time.tv_sec) * 1000000L + (stop_time.tv_usec - start_time.tv_usec);
#endif


#ifdef TIMING
    std::cout << "Total executing time " << total_time << " microseconds\n";
#endif

}
