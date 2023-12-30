CC = gcc
CXX = g++
NVCC = nvcc
MPICC = mpicc
MPICXX = mpicxx


CFLAGS  := -O3
XCFLAGS := -fopenmp
LDFLAGS  := -lm -lpng -lz


CXXFLAGS = -O3 -pthread -march=native -fopenmp
CFLAGS1 = -O3 -lm -pthread -march=native -fopenmp
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61

TARGETS = kmeans kmeans_omp kmeans_hybrid kmeans_cuda
EXES = kmeans kmeans_omp kmeans_hybrid kmeans_cuda

.PHONY: all
all: $(TARGETS) $(EXES)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(EXES)

kmeans: kmeans.cc
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $?

kmeans_omp: kmeans_omp.cc
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $?

kmeans_hybrid: kmeans_hybrid.cc
	$(MPICXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $?

kmeans_cuda: kmeans_cuda.cu
	$(NVCC) $(NVFLAGS) $(CFLAGS) $(LDFLAGS) -Xcompiler="$(XCFLAGS)" -o $@ $?



