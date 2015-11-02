CC=icpc
MPICC=mpiicpc

CFLAGS=-O3 -ipo -xHost -openmp -Wall -Wno-sign-compare

all: omp_lda
	cd ./tools && make all

clean:
	rm -f *.o
	rm -f omp_lda infer
	cd ./tools && make clean

cmd_flags.o: cmd_flags.cc cmd_flags.h
	$(CC) -c $(CFLAGS)  cmd_flags.cc -o cmd_flags.o

common.o: common.cc common.h
	$(CC) -c $(CFLAGS)  common.cc -o common.o

document.o: document.cc document.h common.o
	$(CC) -c $(CFLAGS)  document.cc -o document.o

model.o: model.cc model.h common.o
	$(CC) -c $(CFLAGS)  model.cc -o model.o

accumulative_model.o: accumulative_model.cc accumulative_model.h common.o model.o
	$(CC) -c $(CFLAGS)  accumulative_model.cc -o accumulative_model.o

sampler.o: sampler.cc sampler.h common.o document.o model.o accumulative_model.o
	$(CC) -c $(CFLAGS)  sampler.cc -o sampler.o

omp_lda: omp_lda.cc cmd_flags.o common.o document.o model.o accumulative_model.o sampler.o
	$(MPICC) $(CFLAGS) omp_lda.cc cmd_flags.o common.o document.o model.o accumulative_model.o sampler.o -o omp_lda

