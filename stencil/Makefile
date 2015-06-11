DEFS = -DEDIM=4 -DNI=8 -DNJ=8 -DSTEPS=32 -DSIZE=1
eDEFS = -DMPI_BUF_SIZE=256

CCFLAGS += -O3 $(DEFS)

INCS = -I. -I/usr/local/browndeer/include -I/usr/local/browndeer/include/coprthr
LIBS = -L/usr/local/browndeer/lib -lcoprthr -lcoprthrcc -lm

TARGET = main.x mpi_tfunc.cbin.3.e32

all: $(TARGET)

.PHONY: clean install uninstall $(SUBDIRS)

.SUFFIXES:
.SUFFIXES: .c .o .x .cbin.3.e32

main.x: main.o
	$(CC) -o main.x main.o $(LIBS)

.c.cbin.3.e32:
	clcc --coprthr-cc -mtarget=e32 -D__link_mpi__ --dump-bin -I ./ $(eDEFS) -DCOPRTHR_MPI_COMPAT $<
	chmod a+r $@

.c.o:
	$(CC) $(CCFLAGS) $(INCS) -c $<

clean: $(SUBDIRS)
	rm -f *.o 
	rm -f $(TARGET)

distclean: clean 


