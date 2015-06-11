/* mpi_tfunc.c
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

#include <coprthr_mpi.h>
#define fma(x,y,z) __builtin_fmaf(x,y,z)

typedef struct {
   int ni, nj;
   int di, dj;
   int niter;
   float* A;
   float* B;
   float w0, w1, w2, w3, w4, w5;
} my_args_t;

__kernel void
stencil_thread( void* p )
{
	my_args_t* pargs = (my_args_t*)p;

	int i,j;
	int NI = pargs->ni;
	int NJ = pargs->nj;
	int di = pargs->di;
	int dj = pargs->dj;
	int niter = pargs->niter;
	float* A = pargs->A;
	float* B = pargs->B;
	float w0 = pargs->w0;
	float w1 = pargs->w1;
	float w2 = pargs->w2;
	float w3 = pargs->w3;
	float w4 = pargs->w4;

	int myrank_2d, mycoords[2];
	int dims[2] = {di, dj};
	int periods[2] = {1, 1}; // Periodic communication but ignoring edge copy where irrelvant

	MPI_Status status;
	MPI_Init(0,MPI_BUF_SIZE);

	MPI_Comm comm = MPI_COMM_THREAD;
	MPI_Comm comm_2d;
	MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);
	MPI_Comm_rank(comm_2d, &myrank_2d);
	MPI_Cart_coords(comm_2d, myrank_2d, 2, mycoords);

	int x = mycoords[0];
	int y = mycoords[1];

	// ranks of neighbors
	int north, south, west, east;
	MPI_Cart_shift(comm_2d, 0, 1, &west, &east);
	MPI_Cart_shift(comm_2d, 1, 1, &north, &south);

	// local stencil sizes with padding
	int ni = (NI-2) / di + 2;
	int nj = (NJ-2) / dj + 2;

	// Load the initial values
	void* memfree = coprthr_tls_sbrk(0);
	float* a = (float*)coprthr_tls_sbrk(ni*nj*sizeof(float));
	float* b = (float*)coprthr_tls_sbrk(ni*nj*sizeof(float));
	float* nsbuf = (float*)coprthr_tls_sbrk(ni*sizeof(float));
	float* webuf = (float*)coprthr_tls_sbrk((nj-2)*sizeof(float));
	long long* srcadr;
	long long* dstadr;
	long long* nsend = (long long*)(nsbuf + ni);

	// Copy initial conditions (2D DMA would be better)
	for (j=0; j<nj; j++) e_dma_copy(a+j*ni, A + (y*(ni-2)+j)*NI+x*(nj-2), ni*sizeof(float));

	// Initial conditions
//	if(y==0) for (i=0; i<ni-2; i++) a[i] = -2.0f;
//	if(y==dj) for (i=0; i<ni-2; i++) a[(nj-1)*ni+i] = 1.0f;
//	if(x==di) for (j=0; j<nj-2; j++) a[(j+2)*ni-1] = -1.0f;
//	if(x==0) for (j=0; j<nj-2; j++) a[(j+1)*ni] = 2.0f;

	// Copy "a" into "b" (only need fixed borders would be better)
	for (i=0; i<ni*nj; i++) b[i] = a[i];

	while (niter--) {

/*		for (j=1; j<nj-1; j++) {
			for (i=1; i<ni-1; i++) {
				b[j*ni+i] = w0*a[j*ni+i-1] + w1*a[j*ni+i] + w2*a[j*ni+i+1] + w3*a[j*ni+i-ni] + w4*a[j*ni+i+ni];
			}
		}*/

		for (j=0; j<nj-2; j+=4)
		{
			float a14 = a[(j+1)*ni+0];
			float a15 = a[(j+1)*ni+1];
			float a24 = a[(j+2)*ni+0];
			float a25 = a[(j+2)*ni+1];
			float a34 = a[(j+3)*ni+0];
			float a35 = a[(j+3)*ni+1];
			float a44 = a[(j+4)*ni+0];
			float a45 = a[(j+4)*ni+1];
			for (i=0; i<ni-2; i+=4)
			{
				float a01 = a[(j+0)*ni+i+1];
				float a02 = a[(j+0)*ni+i+2];
				float a03 = a[(j+0)*ni+i+3];
				float a04 = a[(j+0)*ni+i+4];
				float a10 = a14;
				float a11 = a15;
				float a12 = a[(j+1)*ni+i+2];
				float a13 = a[(j+1)*ni+i+3];
				a14 = a[(j+1)*ni+i+4];
				a15 = a[(j+1)*ni+i+5];
				float a20 = a24;
				float a21 = a25;
				float a22 = a[(j+2)*ni+i+2];
				float a23 = a[(j+2)*ni+i+3];
				a24 = a[(j+2)*ni+i+4];
				a25 = a[(j+2)*ni+i+5];
				float a30 = a34;
				float a31 = a35;
				float a32 = a[(j+3)*ni+i+2];
				float a33 = a[(j+3)*ni+i+3];
				a34 = a[(j+3)*ni+i+4];
				a35 = a[(j+3)*ni+i+5];
				float a40 = a44;
				float a41 = a45;
				float a42 = a[(j+4)*ni+i+2];
				float a43 = a[(j+4)*ni+i+3];
				a44 = a[(j+4)*ni+i+4];
				a45 = a[(j+4)*ni+i+5];
				float a51 = a[(j+5)*ni+i+1];
				float a52 = a[(j+5)*ni+i+2];
				float a53 = a[(j+5)*ni+i+3];
				float a54 = a[(j+5)*ni+i+4];

				b[(j+1)*ni+i+1] = fma(w4,a21,fma(w3,a01,fma(w2,a12,fma(w1,a11,w0*a10))));
				b[(j+1)*ni+i+2] = fma(w4,a22,fma(w3,a02,fma(w2,a13,fma(w1,a12,w0*a11))));
				b[(j+1)*ni+i+3] = fma(w4,a23,fma(w3,a03,fma(w2,a14,fma(w1,a13,w0*a12))));
				b[(j+1)*ni+i+4] = fma(w4,a24,fma(w3,a04,fma(w2,a15,fma(w1,a14,w0*a13))));
				b[(j+2)*ni+i+1] = fma(w4,a31,fma(w3,a11,fma(w2,a22,fma(w1,a21,w0*a20))));
				b[(j+2)*ni+i+2] = fma(w4,a32,fma(w3,a12,fma(w2,a23,fma(w1,a22,w0*a21))));
				b[(j+2)*ni+i+3] = fma(w4,a33,fma(w3,a13,fma(w2,a24,fma(w1,a23,w0*a22))));
				b[(j+2)*ni+i+4] = fma(w4,a34,fma(w3,a14,fma(w2,a25,fma(w1,a24,w0*a23))));
				b[(j+3)*ni+i+1] = fma(w4,a41,fma(w3,a21,fma(w2,a32,fma(w1,a31,w0*a30))));
				b[(j+3)*ni+i+2] = fma(w4,a42,fma(w3,a22,fma(w2,a33,fma(w1,a32,w0*a31))));
				b[(j+3)*ni+i+3] = fma(w4,a43,fma(w3,a23,fma(w2,a34,fma(w1,a33,w0*a32))));
				b[(j+3)*ni+i+4] = fma(w4,a44,fma(w3,a24,fma(w2,a35,fma(w1,a34,w0*a33))));
				b[(j+4)*ni+i+1] = fma(w4,a51,fma(w3,a31,fma(w2,a42,fma(w1,a41,w0*a40))));
				b[(j+4)*ni+i+2] = fma(w4,a52,fma(w3,a32,fma(w2,a43,fma(w1,a42,w0*a41))));
				b[(j+4)*ni+i+3] = fma(w4,a53,fma(w3,a33,fma(w2,a44,fma(w1,a43,w0*a42))));
				b[(j+4)*ni+i+4] = fma(w4,a54,fma(w3,a34,fma(w2,a45,fma(w1,a44,w0*a43))));

			}
		}

		// north/south
		dstadr = (long long*)nsbuf;
		srcadr = (long long*)(b+ni);
		while (dstadr != nsend) *dstadr++ = *srcadr++; // second row
		MPI_Sendrecv_replace(nsbuf, ni, MPI_FLOAT, north, 1, south, 1, comm, &status);
		if (y!=dj-1) {
			dstadr = (long long*)(b+(nj-1)*ni);
			srcadr = (long long*)nsbuf;
			while (srcadr != nsend) *dstadr++ = *srcadr++; // last row
		}
		dstadr = (long long*)nsbuf;
		srcadr = (long long*)(b+(nj-2)*ni);
		while (dstadr != nsend) *dstadr++ = *srcadr++; // second to last row
		MPI_Sendrecv_replace(nsbuf, ni, MPI_FLOAT, south, 1, north, 1, comm, &status);
		if (y) {
			dstadr = (long long*)b;
			srcadr = (long long*)nsbuf;
			while (srcadr != nsend) *dstadr++ = *srcadr++; // first row
		}

		// west/east
		for (j=0; j<nj-2; j++) webuf[j] = b[(j+1)*ni+1]; // second column
		MPI_Sendrecv_replace(webuf, nj-2, MPI_FLOAT, west, 1, east, 1, comm, &status);
		if (x!=di-1) for (j=0; j<nj-2; j++) b[(j+2)*ni-1] = webuf[j]; // last column
		for (j=0; j<nj-2; j++) webuf[j] = b[(j+2)*ni-2]; // second to last column
		MPI_Sendrecv_replace(webuf, nj-2, MPI_FLOAT, east, 1, west, 1, comm, &status);
		if (x) for (j=0; j<nj-2; j++) b[(j+1)*ni] = webuf[j]; // first column

		float* tmp = b;
		b = a;
		a = tmp;
	}

	// Copy internal results
	for (j=1; j<nj-1; j++) e_dma_copy(B + (y*(ni-2)+j)*NI+x*(nj-2)+1, a+j*ni+1, (ni-2)*sizeof(float));

	coprthr_tls_brk(memfree);

	MPI_Finalize();
}
