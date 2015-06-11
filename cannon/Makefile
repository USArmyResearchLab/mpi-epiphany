DEFS = -DEDIM=4 -DSIZE=128 -DSIZE2=1 -DSIZE3=1 -DMPI_BUF_SIZE=1536 -DLOOP3=1 -DLOOP2=1 -DLOOP1=1

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
	clcc --coprthr-cc -mtarget=e32 -D__link_mpi__ --dump-bin -I ./ $(DEFS) -DCOPRTHR_MPI_COMPAT $<
	chmod a+r $@

.c.o:
	$(CC) $(CCFLAGS) $(INCS) -c $<

cannon.x: cannon.c
	mpicc -O3 cannon.c -lm -o cannon.x

clean: $(SUBDIRS)
	rm -f *.o 
	rm -f $(TARGET)
	rm -f coprthr_mpi.su

distclean: clean 


