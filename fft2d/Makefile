# Makefile for compiling the MPI 2D FFT code for Epiphany

DEFS = -DEDIM=4 -DNSIZE=128 -DMSIZE=7 -DMPI_BUF_SIZE=512 -DLOOP=1

CCFLAGS += -O2 $(DEFS)

INCS = -I. -I/usr/local/browndeer/coprthr2/include
LIBS = -L/usr/local/browndeer/coprthr2/lib -lcoprthr -lcoprthrcc -lm
ELIBS = -L/usr/local/browndeer/coprthr2/lib -lcoprthr_mpi -lcoprthr2_dev

TARGET = main.x mpi_tfunc.e32 

all: $(TARGET)

.PHONY: clean install uninstall $(SUBDIRS)

.SUFFIXES:
.SUFFIXES: .c .o .x .e32

.c.e32:
	coprcc -I ./ $(DEFS) $(ELIBS) -DCOPRTHR_MPI_COMPAT $< -o $@

main.x: main.o
	$(CC) -o main.x main.o $(LIBS)

.c.o:
	$(CC) $(CCFLAGS) $(INCS) -c $<

clean: $(SUBDIRS)
	rm -f *.o *.x *.e32

distclean: clean 


