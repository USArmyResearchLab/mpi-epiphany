# Makefile for compiling the MPI 2D FFT code for Epiphany

DEFS = -DEDIM=4 -DNSIZE=128 -DMSIZE=7 -DMPI_BUF_SIZE=512 -DLOOP=1

CCFLAGS += -O2 $(DEFS)

INCS = -I. -I/usr/local/browndeer/include -I/usr/local/browndeer/include/coprthr
LIBS = -L/usr/local/browndeer/lib -lcoprthr -lcoprthrcc -lm

TARGET = main.x mpi_tfunc.cbin.3.e32 

all: $(TARGET)

.PHONY: clean install uninstall $(SUBDIRS)

.SUFFIXES:
.SUFFIXES: .c .o .x .cbin.3.e32

.c.cbin.3.e32:
	clcc --coprthr-cc -mtarget=e32 -D__link_mpi__ --dump-bin -I ./ $(DEFS) -DCOPRTHR_MPI_COMPAT $<
	chmod a+r $@

main.x: main.o
	$(CC) -o main.x main.o $(LIBS)

.c.o:
	$(CC) $(CCFLAGS) $(INCS) -c $<

clean: $(SUBDIRS)
	rm -f *.o *.x *.cbin.3.e32

distclean: clean 


