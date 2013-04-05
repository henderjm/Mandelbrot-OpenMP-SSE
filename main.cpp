/*
 * CS3014 Mandelbrot Project
 * 
 * Using techniques we've covered in class, accelerate the rendering of
 * the M set.
 * 
 * Hints
 * 
 * 1) Vectorize
 * 2) Use threads
 * 3) Load Balance
 * 4) Profile and Optimise
 * 
 * Potential FAQ.
 * 
 * Q1) Why when I zoom in far while palying with the code, why does the image begin to render all blocky?
 * A1) In order to render at increasing depths we must use increasingly higher precision floats
 * 	   We quickly run out of precision with 32 bits floats. Change all floats to doubles if you want
 * 	   dive deeper. Eventually you will however run out of precision again and need to integrate an
 * 	   infinite precision math library or use other techniques.
 * 
 * Q2) Why do some frames render much faster than others?
 * A2) Frames with a lot of black, i.e, frames showing a lot of set M, show pixels that ran until the 
 *     maximum number of iterations was reached before bailout. This means more CPU time was consumed
 */



#include <iostream>
#include <cmath>
#include <omp.h>
#include <xmmintrin.h>
#define TIMING
#ifdef TIMING
#include <sys/time.h>
#endif



#include "Screen.h"


/*
 * You can't change these values to accelerate the rendering.
 * Feel free to play with them to render different images though.
 */
const int 	MAX_ITS = 1000;			//Max Iterations before we assume the point will not escape
const int 	HXRES = 700; 			// horizontal resolution	
const int 	HYRES = 700;			// vertical resolution
const int 	MAX_DEPTH = 40;		// max depth of zoom
const float ZOOM_FACTOR = 1.02;		// zoom between each frame

/* Change these to zoom into different parts of the image */
const float PX = -0.7022952861;	// Centre point we'll zoom on - Real component
const float PY = +0.3502203400;	// Imaginary component


/*
 * The palette. Modifying this can produce some really interesting renders.
 * The colours are arranged R1,G1,B1, R2, G2, B2, R3.... etc.
 * RGB values are 0 to 255 with 0 being darkest and 255 brightest
 * 0,0,0 is black
 * 255,255,255 is white
 * 255,0,0 is bright red
 */
unsigned char pal[]={
    255,180,4,
    240,156,4,
    220,124,4,
    156,71,4,
    72,20,4,
    251,180,4,
    180,74,4,
    180,70,4,
    164,91,4,
    100,28,4,
    191,82,4,
    47,5,4,
    138,39,4,
    81,27,4,
    192,89,4,
    61,27,4,
    216,148,4,
    71,14,4,
    142,48,4,
    196,102,4,
    58,9,4,
    132,45,4,
    95,15,4,
    92,21,4,
    166,59,4,
    244,178,4,
    194,121,4,
    120,41,4,
    53,14,4,
    80,15,4,
    23,3,4,
    249,204,4,
    97,25,4,
    124,30,4,
    151,57,4,
    104,36,4,
    239,171,4,
    131,57,4,
    111,23,4,
    4,2,4};
const int PAL_SIZE = 40;  //Number of entries in the palette 



/* 
 * Return true if the point cx,cy is a member of set M.
 * iterations is set to the number of iterations until escape.
 */
bool member(float cx, float cy, int& iterations)
{
    float x = 0.0;
    float y = 0.0;
    iterations = 0;
    while ((x*x + y*y < (2*2)) && (iterations < MAX_ITS)) {
        float xtemp = x*x - y*y + cx;
        y = 2*x*y + cy;
        x = xtemp;
        iterations++;
    }

    return (iterations == MAX_ITS);
}

__m128 sse_member(__m128 cx, __m128 cy) {
   __m128 x = _mm_set1_ps(0.0);
   __m128 y = _mm_set1_ps(0.0);
   __m128 iterations = _mm_set1_ps(0.0);
   __m128 iterations_values;
   __m128 mask = _mm_set1_ps(1.0);
   __m128 temp;
   int iterations_count = 0;

   while( 
   	// basically this goes through the four values in iteration_values
   	// and checks if checks if they are less than 4
   	// we keep going until they are all over 4
	   	(_mm_movemask_ps(
		   	iterations_values =(
		   		// x^2 + y^2 < 4?
		   		_mm_cmplt_ps( 
		   			// x^2 + y^2
		   			_mm_add_ps( _mm_mul_ps(x,x),_mm_mul_ps(y,y)), 
					_mm_set1_ps(4.0) 
				)
			))
	   	)!= 0 
		&& 
		iterations_count < MAX_ITS)
   {
      iterations = _mm_add_ps(iterations, _mm_and_ps(iterations_values, mask)); 

      // temp =  x^2 - y^2 + cx
      temp = _mm_add_ps( _mm_sub_ps( _mm_mul_ps(x,x), _mm_mul_ps(y,y)), cx);
      // y = 2*x*y + cy
      y = _mm_add_ps( _mm_mul_ps( _mm_mul_ps(x,y), _mm_set1_ps(2.0)), cy);

      x = temp;
      iterations_count++;
    
   }

   return iterations;

}

int main()
{	
    int hx, hy, j= 0,i,b;

    float m=1.0; /* initial  magnification		*/
 //   float * arr =  (float *)malloc(sizeof(float) * 4);  // make sure it is alligned
    float arr[4];
        /* Create a screen to render to */
    Screen *screen;
    screen = new Screen(HXRES, HYRES);
    int p = omp_get_max_threads();
    printf("p = %d\n", p);
    
    int depth=0;

#ifdef TIMING
    struct timeval start_time;
    struct timeval stop_time;
    long long total_time = 0;
#endif
    __m128 cx4, cy4;
    __m128 vectorpoint5 = _mm_set1_ps(-0.5);
    __m128 PX4 = _mm_set1_ps(PX);
    __m128 HXRES4;
    __m128 HYRES4 = _mm_set1_ps((float)HYRES);
    HYRES4 = _mm_rcp_ps(HYRES4);
    HXRES4 = _mm_set1_ps((float)HXRES);
    HXRES4 = _mm_rcp_ps(HXRES4);

    while (depth < MAX_DEPTH) {
#ifdef TIMING
        /* record starting time */
        gettimeofday(&start_time, NULL);
#endif
        
        float temp = 4.0f * (1/m); // (4 * (1/m)) == (4 / m)
        __m128 vector4divM = _mm_set1_ps(temp); // 4 / m
        __m128 recipVector4divM = _mm_rcp_ps(vector4divM); // 1 / (4/m)
        __m128 temp2 = _mm_mul_ps(PX4, recipVector4divM); // (PX/(4/m)) ==> (PX * (1 / (4/m))) reciprocal mahem .. very confused now


#pragma omp parallel for private(cx4, cy4, hx, arr, j, b ,i) \
        shared(m, HXRES4)
        for (hy=0; hy<HYRES; hy++) {
            
            //         cy4 = _mm_setr_ps((float)hy,(float)hy+1,(float)hy+2,(float)hy+3); // (float)hx
            //         cy4 = _mm_mul_ps(cx4, HYRES4); // multiplying the reciprocal will speed up calculations -- (float)hx/(float)HXRES
            //         cy4 = _mm_add_ps(cx4, vectorpoint5); //  ((float)hx/(float)HXRES) -0.5)
            //         cy4 = _mm_mul_ps(_mm_add_ps(cy4,temp2), vector4divM);
            float cy = ((((float)hy/(float)HYRES) -0.5 + (PY/(4.0/m)))*(4.0f/m));
            cy4 = _mm_set1_ps(cy);
            
            for (hx=0; hx < HXRES - (HXRES & (0x3)); hx+=4) {
                cx4 = _mm_setr_ps((float)hx,(float)hx+1,(float)hx+2,(float)hx+3); // (float)hx
                /* 
                 * Translate pixel coordinates to complex plane coordinates centred
                 * on PX, PY
                 ***
                 * Look into getting the reciprocal here
                 */

                cx4 = _mm_mul_ps(cx4, HXRES4); // multiplying the reciprocal will speed up calculations -- (float)hx/(float)HXRES
                cx4 = _mm_add_ps(cx4, vectorpoint5); //  ((float)hx/(float)HXRES) -0.5)
                cx4 = _mm_mul_ps(_mm_add_ps(cx4,temp2), vector4divM);
                //             float cx = ((((float)hx/(float)HXRES) -0.5 + (PX/(4.0/m)))*(4.0f/m)); 

                                _mm_store_ps(arr, sse_member(cx4, cy4));
                
                for(j = 0; j < 4 ; j++) {

                    if (arr[j] != MAX_ITS) {
                        i=((int)arr[j]%40) - 1;
                        b = i*3;
                        screen->putpixel(hx+j, hy, pal[b], pal[b+1], pal[b+2]);
                    } else {
                        screen->putpixel(hx+j, hy, 0, 0, 0);
                    }
                                 
                }
            }
        }
#pragma omp barrier
#ifdef TIMING        
        gettimeofday(&stop_time, NULL);
        total_time += (stop_time.tv_sec - start_time.tv_sec) * 1000000L + (stop_time.tv_usec - start_time.tv_usec);
#endif
        /* Show the rendered image on the screen */
        screen->flip();
        std::cout << "Render done " << depth++ <<std::endl;

        /* Zoom in */
        m *= ZOOM_FACTOR;
    }


    sleep(10);
#ifdef TIMING
    std::cout << "Total executing time " << total_time << " microseconds\n";
#endif
    std::cout << "Clean Exit"<< std::endl;

}

