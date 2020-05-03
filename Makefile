OPTFLAG =
LINKFLAGS = -lm -L/usr/lib/cuda/lib64/ -lcudadevrt -lcudart -lstdc++ -L./ -I./


default: mpi-cuda-local 
optlocal: OPTFLAG += -O3
optlocal: mpi-cuda-local
optim: OPTFLAG += -O3
optim: default


mpi-cuda-local: delnetstdpinputmpi.c delnetmpi.c delnetmpi.h spkrcd.c spkrcd.h paramutils.c paramutils.h simkernelsmpicuda.cu simkernelsmpicuda.h simutilsmpi.c simutilsmpi.h cuallocate.cu cuallocate.h
	mpicc -g -Wall $(OPTFLAG) -c delnetstdpinputmpi.c -o delnetstdpinputmpi.o -lm $(LINKFLAGS)
	mpicc -g -Wall $(OPTFLAG) -c delnetmpi.c -o delnetmpi.o $(LINKFLAGS)
	gcc -g -Wall $(OPTFLAG) -c spkrcd.c -o spkrcd.o	-lm
	gcc -g -Wall $(OPTFLAG) -c paramutils.c -o paramutils.o -lm
	nvcc -g -G $(OPTFLAG) -L./ -I./ -lm -c simkernelsmpicuda.cu -o simkernelsmpicuda.o 
	nvcc -g -G $(OPTFLAG) -c cuallocate.cu -o cuallocate.o
	mpicc -g -Wall $(OPTFLAG) -c simutilsmpi.c -o simutilsmpi.o -lm $(LINKFLAGS)
	mpicc -g -Wall $(OPTFLAG) delnetstdpinputmpi.o delnetmpi.o spkrcd.o paramutils.o simkernelsmpicuda.o cuallocate.o simutilsmpi.o -o runtrial-mpicuda $(LINKFLAGS)

clean:
	rm *.o runtrial-mpi*


