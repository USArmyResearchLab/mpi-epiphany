MPI Epiphany Examples
======

The examples below use the threaded MPI model for the Adapteva Epiphany
architecture, specifically the Parallella platform.  All applications are
distributed across the 16 Epiphany cores.

The codes were developed by the US Army Research Laboratory, Computational
Sciences Division, Computing Architectures Branch.

The examples were tested on the 2015.1 Parallella image.

In order to build and test this example, at present you need to download and
install a [pre-release COPRTHR-2 (Anthem)
library](http://www.browndeertechnology.com/anthem.htm). The library is
presently free to download and use for experimentation (non-commercial use).

-JAR Updated 6/10/16

Cannon's Algorithm for Dense Matrix-Matrix Multiply
------

This software demonstrates a high performance implementation of [Cannon's
Algorithm](http://en.wikipedia.org/wiki/Cannon%27s_algorithm) for matrix
multiplication for two-dimensional network meshes.  The communication is
blocking and uses the `MPI_Sendrecv_replace` routine for communication between
cores.

View [./cannon](./cannon) source code or the [README](./cannon/README).

2D Fast Fourier Transform (FFT)
------

This demonstrates a 2D
[FFT](http://en.wikipedia.org/wiki/Fast_Fourier_transform) with a 2D domain
decomposition and uses the `MPI_Sendrecv_replace` for communication

View [./fft2d](./fft2d) source code.

N-body Particle Simulation
------

This example demonstrates a high performance gravitational particle interaction
([N-body simulation](http://en.wikipedia.org/wiki/N-body_simulation)) using
`MPI_Sendrecv_replace` for convenience.

View [./nbody](./nbody) source code or the [README](./nbody/README).

Five-Point 2D Stencil Update
-----

This is a high performance example of a finite difference scheme using a
5-point stencil update where a solution is iterated until convergence and
boundary data is communicated between neighboring cores over the
two-dimensional domain decomposition.

View [./stencil](./stencil) source code.
