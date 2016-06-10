DEFS = -DEDIM=4 -DSIZE=128 -DSIZE2=1 -DSIZE3=1 -DMPI_BUF_SIZE=1536 -DLOOP3=1 -DLOOP2=1 -DLOOP1=1

CCFLAGS += -O3 $(DEFS)

INCS = -I. -I/usr/local/browndeer/coprthr2/include
LIBS = -L/usr/local/browndeer/coprthr2/lib -lcoprthr -lcoprthrcc -lm

ELIBS = -L/usr/local/browndeer/coprthr2/lib -lcoprthr_mpi -lcoprthr2_dev

TARGET = main.x mpi_tfunc.e32

all: $(TARGET)

.PHONY: clean install uninstall $(SUBDIRS)

.SUFFIXES:
.SUFFIXES: .c .o .x .e32

main.x: main.o
	$(CC) -o main.x main.o $(LIBS)

.c.e32:
	coprcc -I ./ $(DEFS) $(ELIBS) -DCOPRTHR_MPI_COMPAT $< -o $@

.c.o:
	$(CC) $(CCFLAGS) $(INCS) -c $<

clean: $(SUBDIRS)
	rm -f *.o 
	rm -f $(TARGET)
	rm -f coprthr_mpi.su

distclean: clean 


