# native-conservation-core Makefile
# Builds C library, CUDA kernels, and benchmarks

CC ?= cc
NVCC ?= nvcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra -Iinclude
CUDAFLAGS = -O3 -arch=sm_86 -Iinclude

.PHONY: all bench cuda alu test clean

all: libconservation.a libternary_alu.a

libconservation.a: src/conservation_core.c
	$(CC) $(CFLAGS) -c src/conservation_core.c -o conservation_core.o
	ar rcs libconservation.a conservation_core.o

libternary_alu.a: src/ternary_alu.c
	$(CC) $(CFLAGS) -c src/ternary_alu.c -o ternary_alu.o
	ar rcs libternary_alu.a ternary_alu.o

bench: src/conservation_core.c
	$(CC) $(CFLAGS) -DBENCHMARK -o benchmarks/conservation_bench src/conservation_core.c -lm

alu: src/ternary_alu.c
	$(CC) $(CFLAGS) -DTEST_ALU -Iinclude -o benchmarks/ternary_alu_test src/ternary_alu.c -lm

cuda: src/ternary_mac_kernel.cu
	$(NVCC) $(CUDAFLAGS) -o benchmarks/ternary_mac src/ternary_mac_kernel.cu

test: bench alu cuda
	./benchmarks/conservation_bench
	./benchmarks/ternary_alu_test
	./benchmarks/ternary_mac

clean:
	rm -f conservation_core.o ternary_alu.o libconservation.a libternary_alu.a
	rm -f benchmarks/conservation_bench benchmarks/ternary_mac benchmarks/ternary_alu_test
