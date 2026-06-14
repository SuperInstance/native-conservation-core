# native-conservation-core Makefile
# Builds C library, CUDA kernels, and benchmarks

CC ?= cc
NVCC ?= nvcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra -Iinclude
CUDAFLAGS = -O3 -arch=sm_86 -Iinclude

.PHONY: all bench cuda clean

all: libconservation.a

libconservation.a: src/conservation_core.c
	$(CC) $(CFLAGS) -c src/conservation_core.c -o conservation_core.o
	ar rcs libconservation.a conservation_core.o

bench: src/conservation_core.c
	$(CC) $(CFLAGS) -DBENCHMARK -o benchmarks/conservation_bench src/conservation_core.c -lm

cuda: src/ternary_mac_kernel.cu
	$(NVCC) $(CUDAFLAGS) -o benchmarks/ternary_mac src/ternary_mac_kernel.cu

clean:
	rm -f conservation_core.o libconservation.a
	rm -f benchmarks/conservation_bench benchmarks/ternary_mac
