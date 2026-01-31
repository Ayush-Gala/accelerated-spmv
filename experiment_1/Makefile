# Compiler settings
CC = cc
MPICC = mpicc
GCC = gcc

# Project paths
ROOT_DIR = ..
INCLUDE_DIR = $(ROOT_DIR)/include

# Compiler flags
CFLAGS = -std=c99 -I$(INCLUDE_DIR) -O3
OMPFLAGS = -fopenmp
DEBUGFLAGS = -D TESTING

# Source files
MMIO = mmio.o
MMIO_SRC = $(ROOT_DIR)/mmio.c
SPMV_SRC = $(ROOT_DIR)/spmv.c
SOURCES = $(SPMV_SRC) spmv-omp.c spmv-mpi.c spmv-hybrid.c

# Executables
EXECS = spmv spmv-omp spmv-mpi spmv-hybrid

# Default target
all: $(EXECS)

# Individual targets
spmv: $(SPMV_SRC) $(MMIO)
	$(CC) $(CFLAGS) -o $@ $(MMIO) $<

spmv-omp: spmv-omp.c $(MMIO)
	$(GCC) $(CFLAGS) -o $@ $< $(MMIO) $(OMPFLAGS)

spmv-mpi: spmv-mpi.c $(MMIO)
	$(MPICC) $(CFLAGS) -o $@ $(MMIO) $<

spmv-hybrid: spmv-hybrid.c $(MMIO)
	$(MPICC) $(CFLAGS) -o $@ $(MMIO) $< $(OMPFLAGS)

$(MMIO): $(MMIO_SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug versions
debug: CFLAGS += $(DEBUGFLAGS)
debug: clean $(EXECS)

# Test target
INPUT ?= $(ROOT_DIR)/example_matrices/simple_4x3.mtx
test: debug
	@echo "Testing sequential version..."
	./spmv $(INPUT)
	@echo "\nTesting OpenMP version..."
	export OMP_NUM_THREADS=2 && ./spmv-omp $(INPUT)
	@echo "\nTesting MPI version..."
	mpirun ./spmv-mpi $(INPUT)
	@echo "\nTesting Hybrid version..."
	export OMP_NUM_THREADS=2 && mpirun ./spmv-hybrid $(INPUT)

# Compare target
compare: debug
	@echo "Comparing outputs..."
	$(ROOT_DIR)/compare.sh test_y seq_y

# Clean target
clean:
	rm -f $(EXECS) *.o
	rm -f test_y
	rm -f seq_y

# Prevent make from confusing the target names with file names
.PHONY: all debug test compare clean