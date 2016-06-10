/* main.c
 *
 * Copyright (c) 2015 Brown Deer Technology, LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *		http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This software was developed by Brown Deer Technology, LLC.
 * For more information contact info@browndeertechnology.com
 */

/* DAR
 * JAR - Minor modifications and definitions 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <complex.h>
typedef float complex cfloat;

#include "coprthr.h"
#include "coprthr_cc.h"
#include "coprthr_thread.h"
#include "coprthr_mpi.h"

#define frand() ((float)rand()/(float)RAND_MAX)
#define __mflops(time,iter,n) ((float)5.0f*(float)n*(float)floor_log2(n)*(float)iter/time/1e6)

unsigned int ones32(unsigned int x) {
	x -= ((x >> 1) & 0x55555555);
	x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
	x = (((x >> 4) + x) & 0x0f0f0f0f);
	x += (x >> 8);
	x += (x >> 16);
	return(x & 0x0000003f);
}

unsigned int floor_log2(unsigned int x) {
	x |= (x >> 1);
	x |= (x >> 2);
	x |= (x >> 4);
	x |= (x >> 8);
	x |= (x >> 16);
	return(ones32(x >> 1));
}

void generate_wn( 
	unsigned int n, unsigned int m, int sign, 
	float* cc, float* ss, cfloat* wn 
) {
	int i;

	float c = cc[m];
	float s = sign * ss[m];

	int n2 = n >> 1;

	wn[0] = 1.0f + 0.0f * I;
	wn[0 + n2] = conj(wn[0]);

	for(i=1; i<n2; i++) {
		wn[i] = (c * crealf(wn[i-1]) - s * cimagf(wn[i-1]))
			+ (s * crealf(wn[i-1]) + c * cimagf(wn[i-1])) * I;
		wn[i + n2] = conj(wn[i]);
	}
}

struct my_args {
	unsigned int n;
	unsigned int m;
	unsigned int l;
	int inverse;
	cfloat* wn;
	cfloat* data2;
};

int main(int argc, char* argv[])
{
	int i,j,k;
	struct timeval t0,t1,t2,t3;

	unsigned int n = NSIZE;
	unsigned int m;
	unsigned int d = EDIM;
	unsigned int l = LOOP;

	i = 1;
	while (i < argc) {
		if (!strcmp(argv[i],"-n")) n = atoi(argv[++i]);
		else if (!strcmp(argv[i],"-d")) d = atoi(argv[++i]);
		else if (!strcmp(argv[i],"-l")) l = atoi(argv[++i]);
		else if (!strcmp(argv[i],"--help") || !strcmp(argv[i],"-h")) goto help;
		else {
			fprintf(stderr,"unrecognized option: %s\n",argv[i]);
			help:
			fprintf(stderr,"use -n [on-chip fft dimension] -d [eCore dimension] -l [artificial loop iterations]\n");
			exit(1);
		}
		++i;
	}

	m = floor_log2(n);
	if((1<<m) != n) {
		printf("Use a power of two for n\n");
		exit(-1);
	}

	printf("Using n=%d, m=%d, d=%d (%d cores), l=%d\n", n, m, d, d*d, l);

	/* open device for threads */
	int dd = coprthr_dopen(COPRTHR_DEVICE_E32,COPRTHR_O_THREAD);
	printf("dd=%d\n",dd); fflush(stdout);
	if (dd<0) {
		printf("device open failed\n");
		exit(-1);
	}

	/* compile thread function */
	char* log = 0;
	coprthr_program_t prg = coprthr_cc_read_bin("./mpi_tfunc.e32",0);
	coprthr_sym_t thr = coprthr_getsym(prg,"fft_thread");
	printf("%p %p\n",prg,thr);

	/* allocate memory on host */
	float* cc = (float*)malloc(16*sizeof(float));
	float* ss = (float*)malloc(16*sizeof(float));
	cfloat* wn_fwd = (cfloat*)malloc(n*sizeof(cfloat));
	cfloat* wn_inv = (cfloat*)malloc(n*sizeof(cfloat));
	cfloat* data1 = (cfloat*)malloc(n*n*sizeof(cfloat));
	cfloat* data2 = (cfloat*)malloc(n*n*sizeof(cfloat));
	cfloat* data3 = (cfloat*)malloc(n*n*sizeof(cfloat));

	/* initialize cos/sin table */
	for(i=0;i<16;i++) {
		cc[i] = (float)cos( 2.0 * M_PI / pow(2.0,(double)i) );
		ss[i] = - (float)sin( 2.0 * M_PI / pow(2.0,(double)i) );
		printf("%2.16f %2.16f\n",cc[i],ss[i]);
	}

	/* initialize wn coefficients */
	generate_wn( n, m, +1, cc, ss, wn_fwd);
	generate_wn( n, m, -1, cc, ss, wn_inv);

	/* initialize data */
	for(i=0; i<n*n; i++) {
		float tmpr = frand();
		float tmpi = frand();
		data1[i] = tmpr + tmpi * I;
		data2[i] = 0.0f;
		data3[i] = 0.0f;
	}

	/* allocate memory on device */
	coprthr_mem_t wn_mem = coprthr_dmalloc(dd,n*sizeof(cfloat),0);
	coprthr_mem_t data2_mem = coprthr_dmalloc(dd,n*n*sizeof(cfloat),0);

	/* copy memory to device */
	coprthr_dwrite(dd,wn_mem,0,wn_fwd,n*sizeof(cfloat),COPRTHR_E_WAIT);
	coprthr_dwrite(dd,data2_mem,0,data1,n*n*sizeof(cfloat),COPRTHR_E_WAIT);

	/* execute parallel calculation */
	struct my_args args_fwd = {
		.n = n, .m = m, .l = l,
		.inverse = 0,
		.wn = coprthr_memptr(wn_mem,0),
		.data2 = coprthr_memptr(data2_mem,0)
	};

	gettimeofday(&t0,0);
	coprthr_mpiexec( dd, d*d, thr, &args_fwd, sizeof(struct my_args),0 );
	gettimeofday(&t1,0);

	/* read back data from memory on device */
	coprthr_dread(dd,data2_mem,0,data2,n*n*sizeof(cfloat),COPRTHR_E_WAIT);

	coprthr_dwrite(dd,wn_mem,0,wn_inv,n*sizeof(cfloat),COPRTHR_E_WAIT);
	coprthr_dwrite(dd,data2_mem,0,data2,n*n*sizeof(cfloat),COPRTHR_E_WAIT);

	/* execute parallel calculation */
	struct my_args args_inv = {
		.n = n, .m = m, .l = l,
		.inverse = 1,
		.wn = coprthr_memptr(wn_mem,0),
		.data2 = coprthr_memptr(data2_mem,0)
	};

	gettimeofday(&t2,0);
	coprthr_mpiexec( dd, d*d, thr, &args_inv, sizeof(struct my_args),0 );
	gettimeofday(&t3,0);

	coprthr_dread(dd,data2_mem,0,data3,n*n*sizeof(cfloat),COPRTHR_E_WAIT);

	if(l==1) {
		for(i=0; i<n*n; i++) {
			printf("%d:\t(%f %f)\t(%f %f)\t(%f %f)\n",i,
				crealf(data1[i]),cimagf(data1[i]),
				crealf(data2[i]),cimagf(data2[i]),
				crealf(data3[i]),cimagf(data3[i]) );
		}
	}
	else printf("Skipping array output, results cannot be correct\n");

	double time_fwd = t1.tv_sec-t0.tv_sec + 1e-6*(t1.tv_usec - t0.tv_usec);
	double time_inv = t3.tv_sec-t2.tv_sec + 1e-6*(t3.tv_usec - t2.tv_usec);
	printf("mpiexec time: forward %f sec inverse %f sec (Loop=%d)\n", time_fwd,time_inv,l);
	printf("Forward Performance: %f MFLOPS\n", __mflops(time_fwd,l,n*n));
	printf("Inverse Performance: %f MFLOPS\n", __mflops(time_inv,l,n*n));

	/* clean up */
	coprthr_dclose(dd);
}
