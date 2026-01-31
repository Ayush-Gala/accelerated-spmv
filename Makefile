# Compiler settings
CC = cc
MPICC = mpicc
GCC = gcc

# Compiler flags
CFLAGS = -std=c99 -I./include/ -O3
OMPFLAGS = -fopenmp
DEBUGFLAGS = -D TESTING

# Source files
MMIO = mmio.o
SOURCES = spmv.c spmv-omp.c spmv-mpi.c spmv-hybrid.c

# Executables
EXECS = spmv spmv-omp spmv-mpi spmv-hybrid

# Default target
all: $(EXECS)

# Individual targets
spmv: spmv.c $(MMIO)
	$(CC) $(CFLAGS) -o $@ $(MMIO) $<

spmv-omp: spmv-omp.c $(MMIO)
	$(GCC) -I./include/ -o $@ $< $(MMIO) $(OMPFLAGS)

spmv-mpi: spmv-mpi.c $(MMIO)
	$(MPICC) $(CFLAGS) -o $@ $(MMIO) $<

spmv-hybrid: spmv-hybrid.c $(MMIO)
	$(MPICC) $(CFLAGS) -o $@ $(MMIO) $< $(OMPFLAGS)

# Debug versions
debug: CFLAGS += $(DEBUGFLAGS)
debug: clean $(EXECS)

# Test target
test: debug
	@echo "Testing sequential version..."
	./spmv input_file
	@echo "\nTesting OpenMP version..."
	export OMP_NUM_THREADS=2 && ./spmv-omp input_file
	@echo "\nTesting MPI version..."
	mpirun ./spmv-mpi input_file
	@echo "\nTesting Hybrid version..."
	export OMP_NUM_THREADS=2 && mpirun ./spmv-hybrid input_file

# Compare target
compare: debug
	@echo "Comparing outputs..."
	./compare.sh test_y seq_y

# Clean target
clean:
	rm -f $(EXECS) *.o
	rm test_y
	rm seq_y

# Prevent make from confusing the target names with file names
.PHONY: all debug test compare clean