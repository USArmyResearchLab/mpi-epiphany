/* main.c
 *
 * Copyright (c) 2015, James A. Ross
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* JAR */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "coprthr.h"
#include "coprthr_cc.h"
#include "coprthr_thread.h"
#include "coprthr_mpi.h"

#define FLOPS_PER_POINT 9
#define __gflops(time,nx,ny,steps) ((float)steps * (float)(nx) * (float)(ny) * FLOPS_PER_POINT / time / 1e9)
#define ERROR(...) do{fprintf(stderr,__VA_ARGS__); exit(-1);}while(0)

/*  __ __ __
	|__|w3|__|
	|w0|w1|w2|
	|__|w4|__|
  5-pt stencil */

typedef struct {
   int ni, nj;
   int di, dj;
   int niter;
   float* A;
   float* B;
   float w0, w1, w2, w3, w4, w5;
} my_args_t;

void print_stencil(float* A, int ni, int nj)
{
	if (ni+nj > 36) return;
	int i,j;
	for (j=0; j<nj; j++) {
		for (i=0; i<ni; i++) {
			printf("% 1.1f ", A[j*ni+i]);
		}
		printf("\n");
	}
	printf("\n");
}

void init_stencil(float* A, float* B, int ni, int nj)
{
	// initialize centers
	int i,j;
	for (j=1; j<nj-1; j++) {
		for (i=1; i<ni-1; i++) {
			int x = j*ni+i;
			A[x] = B[x] = 0.0f;
		}
	}
	// initialize edges
	for(i=1; i<ni-1; i++) {
		A[i] = B[i] = -2.0f; // north
		A[(nj-1)*ni+i] = B[(nj-1)*ni+i] = 1.0f; // south
	}
	for(j=1; j<nj-1; j++) {
		A[j*ni+ni-1] = B[j*ni+ni-1] = -1.0f; // east
		A[j*ni] = B[j*ni] = 2.0f; // west
	}
}

void update_stencil_epiphany(float* A, float* B, int ni, int nj, int di, int dj, int niter, float w0, float w1, float w2, float w3, float w4)
{

	int dd = coprthr_dopen(COPRTHR_DEVICE_E32,COPRTHR_O_THREAD);
	printf("dd=%d\n",dd);
	if (dd<0) ERROR("device open failed\n");

	coprthr_mem_t A_mem = coprthr_dmalloc(dd,ni*nj*sizeof(float),0);
	coprthr_mem_t B_mem = coprthr_dmalloc(dd,ni*nj*sizeof(float),0);

	coprthr_program_t prg = coprthr_cc_read_bin("./mpi_tfunc.cbin.3.e32", 0);
	coprthr_sym_t thr = coprthr_getsym(prg,"stencil_thread");
	printf("prg=%p thr=%p\n",prg,thr);

	coprthr_dwrite(dd,A_mem,0,A,ni*nj*sizeof(float),COPRTHR_E_WAIT);
	coprthr_dwrite(dd,B_mem,0,B,ni*nj*sizeof(float),COPRTHR_E_WAIT); // should really copy this on device

	my_args_t args = {
		.ni = ni, .nj = nj,
		.di = di, .dj = dj,
		.niter = niter,
		.A = coprthr_memptr(A_mem,0),
		.B = coprthr_memptr(B_mem,0),
		.w0 = w0, .w1 = w1, .w2 = w2, .w3 = w3, .w4 = w4
	};

	coprthr_mpiexec(dd, di*dj, thr, &args, sizeof(args),0);

	coprthr_dread(dd,B_mem,0,B,ni*nj*sizeof(float),COPRTHR_E_WAIT);

	print_stencil(B, ni, nj);

}

void update_stencil_cpu(float* A, float* B, int ni, int nj, int niter, float w0, float w1, float w2, float w3, float w4)
{
	// this does not handle edges here, which must be initialized in both A and B
	int i, j, iter = niter;
	while(iter--) {
		for (j=1; j<nj-1; j++) {
			for (i=1; i<ni-1; i++) {
				int x = j*ni+i;
				B[x] = w0*A[x-1] + w1*A[x] + w2*A[x+1] + w3*A[x-ni] + w4*A[x+ni];
			}
		}
		float* tmp = B;
		B = A;
		A = tmp;
	}
	if(niter%2 == 0) for (j=1; j<nj-1; j++) for (i=1; i<ni-1; i++) B[j*ni+i] = A[j*ni+i];
}

int validate_stencil(float* B_host, float* B_device, int ni, int nj)
{
	int errors = 0;
	int i, j;
	float tol = 0.0001;
	for (j=1; j<nj-1; j++) {
		for (i=1; i<ni-1; i++) {
			float diff = fabs(B_host[j*ni+i] - B_device[j*ni+i]);
			if(diff > tol) errors++;
		}
	}
	return errors;
}

int main(int argc, char** argv)
{
	int i,j,k;
	struct timeval t0,t1;
	double time;
	int validate = 0;

	int ni = NI, nj = NJ;
	int niter = STEPS;
	int di = EDIM, dj = EDIM;

	// stencil weights
	float w0 = 0.166666666f;
	float w1 = 0.333333333f;
	float w2 = 0.166666666f;
	float w3 = 0.166666666f;
	float w4 = 0.166666666f;

	i = 1;
	while (i < argc) {
		if (!strcmp(argv[i],"-n")) { ni = atoi(argv[++i]); nj = atoi(argv[++i]); }
		else if (!strcmp(argv[i],"-i")) niter = atoi(argv[++i]);
		else if (!strcmp(argv[i],"-d")) { di = atoi(argv[++i]), dj = atoi(argv[++i]); }
		else if (!strcmp(argv[i],"--validate")) validate = 1;
		else if (!strcmp(argv[i],"--help") || !strcmp(argv[i],"-h")) goto help;
		else {
			fprintf(stderr,"unrecognized option: %s\n",argv[i]);
			help:
			ERROR("use -n [internal X size] [internal Y size] -i [iteration step count] -d [number of Epiphany cores/threads] --validate\n");
		}
		++i;
	}
	if(ni%di) ERROR("ni = %d is not divisible by di = %d\n", ni, di);
	if(nj%dj) ERROR("nj = %d is not divisible by dj = %d\n", nj, dj);

	printf("Using N = {%d x %d}, # iterations = %d, # cores = {%d x %d} = %d\n", ni, nj, niter, di, dj, di*dj);

	// From here on, we're using padded ni and nj values
	ni += 2;
	nj += 2;

	// allocate memory on host
	float* A = (float*)malloc(ni*nj*sizeof(float));
	float* B = (float*)malloc(ni*nj*sizeof(float));

	// initialize
	init_stencil(A, B, ni, nj);

	gettimeofday(&t0,0);
	update_stencil_epiphany(A, B, ni, nj, di, dj, niter, w0, w1, w2, w3, w4);
	gettimeofday(&t1,0);

	time = t1.tv_sec - t0.tv_sec + 1e-6*(t1.tv_usec - t0.tv_usec);
	float gflops = __gflops(time, ni-2, nj-2, niter);
	printf("Epiphany Performance.... : %f GFLOPS (includes overhead)\n",gflops);
	printf("Execution Time.......... : %f seconds\n",time);

	if (validate) {

		float* A_validate = (float*)malloc(ni*nj*sizeof(float));
		float* B_validate = (float*)malloc(ni*nj*sizeof(float));
		init_stencil(A_validate, B_validate, ni, nj);

		printf("Validating on CPU host....\n");
		gettimeofday(&t0,0);
		update_stencil_cpu(A_validate, B_validate, ni, nj, niter, w0, w1, w2, w3, w4);
		gettimeofday(&t1,0);

		int errors = validate_stencil(B, B_validate, ni, nj);

		print_stencil(B_validate, ni, nj);

		time = t1.tv_sec - t0.tv_sec + 1e-6*(t1.tv_usec - t0.tv_usec);
		printf("CPU Execution time... : %f seconds\n",time);
		printf("Errors............... : %d (%0.1f%%)\n", errors, 100.0f*errors/((ni-2)*(nj-2)));
	}

	return 0;

}
