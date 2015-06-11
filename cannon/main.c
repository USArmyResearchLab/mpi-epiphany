/* main.c
 *
 * Copyright (c) 2015, James A. Ross
 * Copyright (c) 2015, Brown Deer Technology, LLC.
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
/* DAR */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#include "coprthr.h"
#include "coprthr_cc.h"
#include "coprthr_thread.h"
#include "coprthr_mpi.h"

float f2(int n) {
	float x = (float)n;
	return (x*(x+1.0f)*(2.0f*x+1.0f)/6.0f);
}

float f3(int n) {
	float x = (float)n;
	float tmp = x*(x+1.0f)/2.0f;
	return (tmp*tmp);
}

float f4(int n) {
	float x = (float)n;
	return (x*(x+1.0f)*(2.0f*x+1.0f)*(3.0f*x*x+3.0f*x-1.0f)/30.0f);
}

struct my_args { int n; int s; int d; float* ga; float* gb; float* gc; };

int main(int argc, char* argv[])
{
	int i,j,k;
	int row;
	struct timeval t0,t1;
	double time = 0.0;
	int n = SIZE;
	int s = SIZE2;
	int s2 = SIZE3;
	int d = EDIM;
	int v = 0;

	i = 1;
	while (i < argc) {
		if (!strcmp(argv[i],"-n")) n = atoi(argv[++i]);
		else if (!strcmp(argv[i],"-s")) s = atoi(argv[++i]);
		else if (!strcmp(argv[i],"-s2")) s2 = atoi(argv[++i]);
		else if (!strcmp(argv[i],"-d")) d = atoi(argv[++i]);
		else if (!strcmp(argv[i],"-v")) v = 1;
		else if (!strcmp(argv[i],"--help") || !strcmp(argv[i],"-h")) goto help;
		else {
			fprintf(stderr,"unrecognized option: %s\n",argv[i]);
			help:
			fprintf(stderr,"use -n [on-chip matrix dimension] -s [off-chip scale factor] -s2 [host scale factor] -d [eCore dimension]\n");
			exit(1);
		}
		++i;
	}

	printf("Using -n=%d, -s=%d, -s2=%d, -d=%d\n", n, s, s2, d);
	int N = s*s2*n;
	int nn = n*n;
	int ss = s*s;

	/* open device for threads */
	int dd = coprthr_dopen(COPRTHR_DEVICE_E32,COPRTHR_O_THREAD);
	printf("dd=%d\n",dd); fflush(stdout);
	if (dd<0) {
		printf("device open failed\n");
		exit(-1);
	}

	/* compile thread function */
	char* log = 0;
	coprthr_program_t prg = coprthr_cc_read_bin("./mpi_tfunc.cbin.3.e32",0);
	coprthr_sym_t thr = coprthr_getsym(prg,"my_thread");
	printf("%p %p\n",prg,thr);

	/* allocate host memory */
	float* ga = (float*)malloc(N*N*sizeof(float));
	float* gb = (float*)malloc(N*N*sizeof(float));
	float* gc = (float*)malloc(N*N*sizeof(float));

	/* allocate memory on device and write a value */
	size_t size_device_mem = s*s*n*n*sizeof(float);
	coprthr_mem_t ga_mem = coprthr_dmalloc(dd,size_device_mem,0);
	coprthr_mem_t gb_mem = coprthr_dmalloc(dd,size_device_mem,0);
	coprthr_mem_t gc_mem = coprthr_dmalloc(dd,size_device_mem,0);

	float* tmpa = (float*)malloc(size_device_mem);
	float* tmpb = (float*)malloc(size_device_mem);
	float* tmpc = (float*)malloc(size_device_mem);
	
	struct my_args args = {
		.n = n, .s = s, .d = d,
		.ga = coprthr_memptr(ga_mem,0),
		.gb = coprthr_memptr(gb_mem,0),
		.gc = coprthr_memptr(gc_mem,0)};

	/* initialize A, B, and C matrices */
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			float x = (float)i + 1.0f;
			float y = (float)j + 1.0f;
			ga[i*N + j] = (float)(x*y*y - 2.0f*x*x*y);
			gb[i*N + j] = (float)(x*y*y + 2.0f*x*x*y);
			gc[i*N + j] = 0.0f;
		}
	}

	gettimeofday(&t0,0);

	for(i=0;i<s2;i++) {
		for(j=0;j<s2;j++) {
			/* write data from host to device */
			size_t row_size = s*n*sizeof(float);
			int nrows = s*n;
			for(row=0;row<nrows;row++) {
				memcpy(tmpc + row*n*s, gc + ((i*nrows+row)*s2 + j)*nrows, row_size);
			}
			coprthr_dwrite(dd, gc_mem, 0, tmpc, nrows*row_size, COPRTHR_E_WAIT);
			for(k=0;k<s2;k++) {
				for(row=0;row<nrows;row++) {
					memcpy(tmpa + row*n*s, ga + ((i*nrows+row)*s2 + k)*nrows, row_size);
					memcpy(tmpb + row*n*s, gb + ((k*nrows+row)*s2 + j)*nrows, row_size);
				}
				coprthr_dwrite(dd, ga_mem, 0, tmpa, nrows*row_size, COPRTHR_E_WAIT);
				coprthr_dwrite(dd, gb_mem, 0, tmpb, nrows*row_size, COPRTHR_E_WAIT);
				/* execute thread function */
				coprthr_mpiexec( dd, d*d, thr, &args, sizeof(struct my_args),0 );
			}
			coprthr_dread(dd, gc_mem, 0, tmpc, nrows*row_size, COPRTHR_E_WAIT);
			for(row=0;row<nrows;row++) {
				memcpy(gc + ((i*nrows+row)*s2 + j)*nrows, tmpc + row*n*s, row_size);
			}
		}
	}

	gettimeofday(&t1,0);
	time += t1.tv_sec - t0.tv_sec + 1e-6*(t1.tv_usec - t0.tv_usec);

	int errors = 0;
	for(i=0;i<N;i++) {
		for(j=0;j<N;j++) {
			float x = (float)i + 1.0f;
			float y = (float)j + 1.0f;
			float d = (2.0f*x*y*f4(N) + (x*y*y-4.0f*x*x*y)*f3(N) - 2.0f*x*x*y*y*f2(N));
			if (v) printf("[%d,%d]: %f %f %f %f\n",i,j, ga[i*N+j], gb[i*N+j], gc[i*N+j], d);
			float e = gc[i*N+j];
			float diff = fabsf(d/e - 1.0f);
			if(diff > 0.01f || isnan(e)) errors++;
		}
	}
	printf("mpiexec time %f sec\n",time);
	printf("# errors: %d\n", errors);

	/* clean up */
	coprthr_dclose(dd);
}
