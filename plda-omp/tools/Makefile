CC=icpc
CFLAGS=-O3

all: pre post

clean:
	rm -f preprocess postprocess

pre: preprocess.cc
	$(CC) $(CFLAGS) -o preprocess preprocess.cc

post: postprocess.cc
	$(CC) $(CFLAGS) -o postprocess postprocess.cc
