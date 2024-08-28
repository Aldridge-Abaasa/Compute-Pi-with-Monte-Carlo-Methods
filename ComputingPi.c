/*****************************************************************************
* DESCRIPTION:
* This program on Computing Pi using the monte carlo simulation.
* using OPENMP.
* Compile: 
*	$gcc ComputingPi.c -o Q2 -fopenmp -lm
* Run:
* 	$./Q2
******************************************************************************/



/*
 * * This code generates random numbers between 0 and 1.
 * * This code is not threadsafe. Below is an example of
 * * how to call ran2.
 * *
 * * float x;
 * * long seed;
 * * x = ran2(&seed);
 * 
 * 
 * 
 * 
 * run file : gcc -fopenmp -o .\Assignment1X.exe .\ComputingPi.c -lm
 * */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/* REMOVED  For this ran2 function, the multiple threads do not need to simultaneously access the shared variables 
'idum2', 'iy' and 'iv'.
Using the OpenMP "threadprivate" directive to make these variables private to each thread. 
The threadprivate directive specifies that each thread has its own copy of a given variable.*/

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

// #pragma omp threadprivate(idum2, iy, iv)

float ran2(long *idum) {
	int j;
	long k;
	static long idum2=123456789;
	static long iy=0;
	static long iv[NTAB];
	float temp;


	#pragma omp critical /*need to ensure that the function's shared variables are accessed and updated by only one thread at a time. One simple way to achieve this is to use a critical section. */
	{
	if (*idum <= 0) {
		if (-(*idum) < 1) *idum=1;
		else *idum = -(*idum);
		idum2=(*idum);
		#pragma omp parallel for  /*added an OpenMP pragma for the for loop to parallelize it. The pragma #pragma omp parallel for instructs the compiler to distribute the loop iterations among the available threads. The loop iterations are partitioned statically among the threads by default.*/
		for (j=NTAB+7;j>=0;j--) {
			k=(*idum)/IQ1;
			*idum=IA1*(*idum-k*IQ1)-k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
			}
		iy=iv[0];
		}
	}
	k=(*idum)/IQ1;
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	if (*idum < 0) *idum += IM1;
	k=idum2/IQ2;
	idum2=IA2*(idum2-k*IQ2)-k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j=iy/NDIV;
	iy=iv[j]-idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}/* (C) Copr. 1986-92 Numerical Recipes Software "!15L1. */  

int main(){
	float x;
	float y;
	float d;
	float pi;
	int nt;
	int nd = 0;
	double start_time, end_time, elapsed_time;

	printf("Enter the number of random points to generate: ");
    scanf("%d", &nt);


	start_time = omp_get_wtime(); // Start the timer
	int i;
	long seed = time(NULL);

	//#pragma omp parallel for/*private(x,y,d) reduction(+:nd)  /*parallelizes the loop using OpenMP, where each thread has its own copies of x, y, and d, and the nd variable is shared among threads and updated using a reduction operation. This allows the loop to be executed much faster since multiple threads can work on different parts of the loop in parallel.*/
	for (i = 0; i < nt; i++) {
		x = ran2(&seed);
        y = ran2(&seed);
        d = x * x + y * y;
        if (d <= 1.0) {
			// #pragma omp atomic
			nd++;
		}
	}
    pi = 4.0 * nd / nt;
	end_time = omp_get_wtime(); // Stop the timer
    elapsed_time = end_time - start_time; // Calculate the elapsed time
    printf("Estimate of pi: %f\n", pi); // Calculate pi 
    printf("Elapsed time: %f seconds\n", elapsed_time); // print out the time it took
	return 0;
}
/*Output without ran2 function being thread safe: 

at nt = 100:
Estimate of pi: 3.440000
Elapsed time: 0.001000 seconds

at nt = 1000: 
Estimate of pi: 3.152000
Elapsed time: 0.000000 seconds

at nt = 10000:
Estimate of pi: 3.127200
Elapsed time: 0.001000 seconds

at nt = 100000:
Estimate of pi: 3.140240
Elapsed time: 0.009000 seconds

at nt = 1000000: 
Estimate of pi: 3.141616
Elapsed time: 0.068000 seconds

at nt = 10000000
Estimate of pi: 3.141462
Elapsed time: 0.591000 seconds

________________________________________________________________________________________________________________
With the ran2 OPENMP constructs: 

nt at 100 :
Estimate of pi: 3.280000
Elapsed time: 0.000000 seconds

nt at 1000:

Estimate of pi: 3.084000
Elapsed time: 0.000000 seconds

nt at 10000:

Estimate of pi: 3.159200
Elapsed time: 0.002000 seconds

nt at 100000
Estimate of pi: 3.134400
Elapsed time: 0.012000 seconds

nt at 1000000
Estimate of pi: 3.142984
Elapsed time: 0.093000 seconds

nt at 10000000
Estimate of pi: 3.141711
Elapsed time: 0.811000 seconds

Since nd changes as its executed, when the program is not thread-safe,  the result depends on the number of threads because the race conditions that occur when multiple threads update the shared variable nd can cause different values of nd to be accumulated. Therefore, to ensure the correctness of the program, thread safety mechanisms such as atomic updates or critical sections should be used to prevent race conditions.
*/