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

// This function performs a serial matrix-matrix multiplication c = a * b
void MatrixMultiply(int n, float *a, float *b, float *c);

typedef struct { int N; int s; int d; float* ga; float* gb; float* gc;} my_args_t;

void __entry
my_thread( void* p) {

	my_args_t* pargs = (my_args_t*)p;

	int N = pargs->N, s = pargs->s, d = pargs->d; 
	float *ga = pargs->ga, *gb = pargs->gb, *gc = pargs->gc;
	int n = N/d;

	int myrank_2d, mycoords[2];
	int dims[2] = {d, d};
	int periods[2] = {1, 1};

	MPI_Status status;
	MPI_Init(0,MPI_BUF_SIZE);

	MPI_Comm comm = MPI_COMM_THREAD;
	MPI_Comm comm_2d;
	MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);
	MPI_Comm_rank(comm_2d, &myrank_2d);
	MPI_Cart_coords(comm_2d, myrank_2d, 2, mycoords);
	// Compute ranks of the up and left shifts
	int uprank, downrank, leftrank, rightrank;
	MPI_Cart_shift(comm_2d, 0, 1, &leftrank, &rightrank);
	MPI_Cart_shift(comm_2d, 1, 1, &uprank, &downrank);

	int x = mycoords[0];
	int y = mycoords[1];
	// this removes initial skew shift by reading in directly
	int skew = (x+y) % d;

	void* memfree = coprthr_tls_sbrk(0);
	float* a = (float*)coprthr_tls_sbrk(n*n*sizeof(float));
	float* b = (float*)coprthr_tls_sbrk(n*n*sizeof(float));
	float* c = (float*)coprthr_tls_sbrk(n*n*sizeof(float));

	e_dma_desc_t dma_c_read, dma_c_write, dma_a_read, dma_b_read;

#define DWORD_WRITE(desc,w,h,W,src,dst) \
	e_dma_set_desc(E_DMA_0, (E_DMA_ENABLE|E_DMA_MASTER|E_DMA_DWORD), 0x0000, \
	0x0008, 0x0008, \
	w/2, h, \
	8, 4*(W-w+2), \
	(void*)src, (void*)dst, &desc)
#define DWORD_READ(desc,w,h,W,src,dst) \
	e_dma_set_desc(E_DMA_0, (E_DMA_ENABLE|E_DMA_MASTER|E_DMA_DWORD), 0x0000, \
	0x0008, 0x0008, \
	w/2, h, \
	4*(W-w+2), 8, \
	(void*)src, (void*)dst, &desc)

int loop;
for(loop=0;loop<LOOP1;loop++) {

	int i,j,k,l;
	for (i=0; i<s; i++) {
		for (j=0; j<s; j++) {
			float* rgc = gc + ((i*N + y*n)*s + j)*N + x*n;
			DWORD_WRITE(dma_c_write,n,n,s*N,c,rgc);
			DWORD_READ(dma_c_read,n,n,s*N,rgc,c);
			// read C
			e_dma_start(&dma_c_read, E_DMA_0);
			e_dma_wait(E_DMA_0);
			for (k=0; k<s; k++) {
				float* rga = ga + ((i*N + y*n)*s + k)*N + skew*n;
				float* rgb = gb + ((k*N + skew*n)*s + j)*N + x*n;
				// read A and B
				DWORD_READ(dma_b_read,n,n,s*N,rgb,b);
				DWORD_READ(dma_a_read,n,n,s*N,rga,a);
				e_dma_start(&dma_b_read, E_DMA_0);
				e_dma_wait(E_DMA_0);
				e_dma_start(&dma_a_read, E_DMA_0);
				e_dma_wait(E_DMA_0);
				// transpose B
				int ji, ii;
				for (ji=0; ji<n-1; ji++) {
					for(ii=ji+1; ii<n; ii++) {
						int tmp = b[ji*n+ii];
						b[ji*n+ii] = b[ii*n+ji];
						b[ii*n+ji] = tmp;
					}
				}
				int loop;
				for (loop=0;loop<LOOP3;loop++) {
				// Get into the main computation loop
				for (l=1; l<d; l++) {
					int loop;
					for(loop=0;loop<LOOP2;loop++)
					MatrixMultiply(n, a, b, c);
					// Shift matrix a left by one and shift matrix b up by one
					MPI_Sendrecv_replace(a, n*n, MPI_FLOAT, leftrank, 1, rightrank, 1, comm_2d, &status);
					MPI_Sendrecv_replace(b, n*n, MPI_FLOAT, uprank, 1, downrank, 1, comm_2d, &status);
				}
				MatrixMultiply(n, a, b, c);
			} // end LOOP3
			}
			// write C
			e_dma_start(&dma_c_write, E_DMA_1);
			e_dma_wait(E_DMA_1);
		}
	}
} // end LOOP1

	coprthr_tls_brk(memfree);

	MPI_Finalize();

}

void MatrixMultiply(int n, float *a, float *b, float *c)
{
	int i, j, k;
	for (i=0; i<n; i+=4) {
		for (j=0; j<n; j+=4) {
			float c00 = c[(i+0)*n+j+0];
			float c10 = c[(i+0)*n+j+1];
			float c20 = c[(i+0)*n+j+2];
			float c30 = c[(i+0)*n+j+3];
			float c01 = c[(i+1)*n+j+0];
			float c11 = c[(i+1)*n+j+1];
			float c21 = c[(i+1)*n+j+2];
			float c31 = c[(i+1)*n+j+3];
			float c02 = c[(i+2)*n+j+0];
			float c12 = c[(i+2)*n+j+1];
			float c22 = c[(i+2)*n+j+2];
			float c32 = c[(i+2)*n+j+3];
			float c03 = c[(i+3)*n+j+0];
			float c13 = c[(i+3)*n+j+1];
			float c23 = c[(i+3)*n+j+2];
			float c33 = c[(i+3)*n+j+3];
			for (k=0; k<n; k+=4) {
				float* a0 = a+(i+0)*n+k;
				float* a1 = a+(i+1)*n+k;
				float* a2 = a+(i+2)*n+k;
				float* a3 = a+(i+3)*n+k;
				float* b0 = b+(j+0)*n+k;
				float* b1 = b+(j+1)*n+k;
				float* b2 = b+(j+2)*n+k;
				float* b3 = b+(j+3)*n+k;
#define fma(x,y,z) __builtin_fmaf(x,y,z)
				c00 = fma(a0[3], b0[3], fma(a0[2], b0[2], fma(a0[1], b0[1], fma(a0[0], b0[0], c00))));
				c01 = fma(a1[3], b0[3], fma(a1[2], b0[2], fma(a1[1], b0[1], fma(a1[0], b0[0], c01))));
				c02 = fma(a2[3], b0[3], fma(a2[2], b0[2], fma(a2[1], b0[1], fma(a2[0], b0[0], c02))));
				c03 = fma(a3[3], b0[3], fma(a3[2], b0[2], fma(a3[1], b0[1], fma(a3[0], b0[0], c03))));
				c10 = fma(a0[3], b1[3], fma(a0[2], b1[2], fma(a0[1], b1[1], fma(a0[0], b1[0], c10))));
				c11 = fma(a1[3], b1[3], fma(a1[2], b1[2], fma(a1[1], b1[1], fma(a1[0], b1[0], c11))));
				c12 = fma(a2[3], b1[3], fma(a2[2], b1[2], fma(a2[1], b1[1], fma(a2[0], b1[0], c12))));
				c13 = fma(a3[3], b1[3], fma(a3[2], b1[2], fma(a3[1], b1[1], fma(a3[0], b1[0], c13))));
				c20 = fma(a0[3], b2[3], fma(a0[2], b2[2], fma(a0[1], b2[1], fma(a0[0], b2[0], c20))));
				c21 = fma(a1[3], b2[3], fma(a1[2], b2[2], fma(a1[1], b2[1], fma(a1[0], b2[0], c21))));
				c22 = fma(a2[3], b2[3], fma(a2[2], b2[2], fma(a2[1], b2[1], fma(a2[0], b2[0], c22))));
				c23 = fma(a3[3], b2[3], fma(a3[2], b2[2], fma(a3[1], b2[1], fma(a3[0], b2[0], c23))));
				c30 = fma(a0[3], b3[3], fma(a0[2], b3[2], fma(a0[1], b3[1], fma(a0[0], b3[0], c30))));
				c31 = fma(a1[3], b3[3], fma(a1[2], b3[2], fma(a1[1], b3[1], fma(a1[0], b3[0], c31))));
				c32 = fma(a2[3], b3[3], fma(a2[2], b3[2], fma(a2[1], b3[1], fma(a2[0], b3[0], c32))));
				c33 = fma(a3[3], b3[3], fma(a3[2], b3[2], fma(a3[1], b3[1], fma(a3[0], b3[0], c33))));
			}
			c[(i+0)*n+j+0] = c00;
			c[(i+0)*n+j+1] = c10;
			c[(i+0)*n+j+2] = c20;
			c[(i+0)*n+j+3] = c30;
			c[(i+1)*n+j+0] = c01;
			c[(i+1)*n+j+1] = c11;
			c[(i+1)*n+j+2] = c21;
			c[(i+1)*n+j+3] = c31;
			c[(i+2)*n+j+0] = c02;
			c[(i+2)*n+j+1] = c12;
			c[(i+2)*n+j+2] = c22;
			c[(i+2)*n+j+3] = c32;
			c[(i+3)*n+j+0] = c03;
			c[(i+3)*n+j+1] = c13;
			c[(i+3)*n+j+2] = c23;
			c[(i+3)*n+j+3] = c33;
		}
	}
}
