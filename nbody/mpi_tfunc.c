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
#include "nbody.h"

void __entry
nbody_thread( void* p )
{
	my_args_t* pargs = (my_args_t*)p;

	int n = pargs->n;
	int cnt = pargs->cnt;
	float dt = pargs->dt;
	float es = pargs->es;
	Particle *particles = pargs->p;
	ParticleV *state = pargs->v;

	int rank, size, npart, i;
	int left, right;

	MPI_Status status;
	MPI_Init(0,MPI_BUF_SIZE);
	MPI_Comm comm = MPI_COMM_THREAD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	MPI_Cart_shift(comm, 0, 1, &left, &right);

	npart = n / size;

	// Load the initial values
	void* memfree = coprthr_tls_sbrk(0);
	Particle* my_particles = (Particle*)coprthr_tls_sbrk(npart*sizeof(Particle));
	ParticleV* my_state = (ParticleV*)coprthr_tls_sbrk(npart*sizeof(ParticleV));
	Particle* sendbuf = (Particle*)coprthr_tls_sbrk(npart*sizeof(Particle));

	e_dma_copy(my_particles, particles + npart*rank, npart*sizeof(Particle));
	e_dma_copy(my_state, state + npart*rank, npart*sizeof(ParticleV));

	while (cnt--) {

		// Load the initial sendbuffer
		for (i=0; i<npart; i++) sendbuf[i] = my_particles[i];

		for (i=0; i<size; i++) {
			// Shift work particles except first
			if (i) MPI_Sendrecv_replace(sendbuf, sizeof(Particle)/sizeof(float)*npart, MPI_FLOAT, left, 1, right, 1, comm, &status);
			// Compute acclerations
			ComputeAccel(my_particles, sendbuf, my_state, npart, es);
		}
		// Once we have the acclerations, we compute the changes in positions and velocities
		ComputeNewPos(my_particles, my_state, npart, dt);
	}

	e_dma_copy(particles + npart*rank, my_particles, npart*sizeof(Particle));
	e_dma_copy(state + npart*rank, my_state, npart*sizeof(ParticleV));

	coprthr_tls_brk(memfree);

	MPI_Finalize();
}
