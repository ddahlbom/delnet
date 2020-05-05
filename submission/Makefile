# Sorry, this is a very dumb makefile
OPTFLAG =
LINKFLAGSLOCAL = -lm -L/usr/lib/cuda/lib64/ -lcudadevrt -lcudart -lstdc++ -L./ -I./
LINKFLAGSAIMOS = -lm -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++


default: aimos 

opt: OPTFLAG += -O3
opt: aimos 
optlocal: OPTFLAG += -O3
optlocal: local 
optlocalnocuda: OPTFLAG += -O3
optlocalnocuda: localnocuda
nocuda: aimosnocuda
optnocuda: OPTFLAG += -O3
optnocuda: aimosnocuda


local: delnetstdpinputmpi.c delnetmpi.c delnetmpi.h spkrcd.c spkrcd.h paramutils.c paramutils.h simkernelsmpicuda.cu simkernelsmpicuda.h simutilsmpicuda.c simutilsmpi.h cuallocate.cu cuallocate.h
	mpicc -g -Wall $(OPTFLAG) -c delnetstdpinputmpi.c -o delnetstdpinputmpi.o
	mpicc -g -Wall $(OPTFLAG) -c delnetmpi.c -o delnetmpi.o 
	gcc -g -Wall $(OPTFLAG) -c spkrcd.c -o spkrcd.o	
	gcc -g -Wall $(OPTFLAG) -c paramutils.c -o paramutils.o
	nvcc -g -G -c $(OPTFLAG) simkernelsmpicuda.cu -o simkernelsmpicuda.o -lm
	nvcc -g -G -c $(OPTFLAG) cuallocate.cu -o cuallocate.o
	mpicc -g -c simutilsmpicuda.c -o simutilsmpicuda.o -lm $(LINKFLAGS) 	# optimization breaks this!!
	mpicc -g -Wall $(OPTFLAG) delnetstdpinputmpi.o delnetmpi.o spkrcd.o paramutils.o simkernelsmpicuda.o cuallocate.o simutilsmpicuda.o -o runtrial-mpi $(LINKFLAGSLOCAL)

localnocuda: delnetstdpinputmpi.c delnetmpi.c delnetmpi.h spkrcd.c spkrcd.h paramutils.c paramutils.h simkernelsmpicuda.cu simkernelsmpicuda.h simutilsmpi.c simutilsmpi.h cuallocate.cu cuallocate.h
	mpicc -g -Wall $(OPTFLAG) -c delnetstdpinputmpi.c -o delnetstdpinputmpi.o
	mpicc -g -Wall $(OPTFLAG) -c delnetmpi.c -o delnetmpi.o 
	gcc -g -Wall $(OPTFLAG) -c spkrcd.c -o spkrcd.o	
	gcc -g -Wall $(OPTFLAG) -c paramutils.c -o paramutils.o
	nvcc -g -G -c $(OPTFLAG) simkernelsmpicuda.cu -o simkernelsmpicuda.o -lm
	nvcc -g -G -c $(OPTFLAG) cuallocate.cu -o cuallocate.o
	mpicc -g -Wall -c simutilsmpi.c -o simutilsmpi.o -lm $(LINKFLAGS) 	# optimization breaks this!!
	mpicc -g -Wall $(OPTFLAG) delnetstdpinputmpi.o delnetmpi.o spkrcd.o paramutils.o simkernelsmpicuda.o cuallocate.o simutilsmpi.o -o runtrial-mpi $(LINKFLAGSLOCAL)

aimos: delnetstdpinputmpi.c delnetmpi.c delnetmpi.h spkrcd.c spkrcd.h paramutils.c paramutils.h simkernelsmpicuda.cu simkernelsmpicuda.h simutilsmpicuda.c simutilsmpi.h cuallocate.cu cuallocate.h
	mpicc -g -Wall $(OPTFLAG) -c delnetstdpinputmpi.c -o delnetstdpinputmpi.o
	mpicc -g -Wall $(OPTFLAG) -c delnetmpi.c -o delnetmpi.o 
	gcc -g -Wall $(OPTFLAG) -c spkrcd.c -o spkrcd.o	
	gcc -g -Wall $(OPTFLAG) -c paramutils.c -o paramutils.o
	nvcc -g -G -c -arch=sm_70 $(OPTFLAG) simkernelsmpicuda.cu -o simkernelsmpicuda.o -lm
	nvcc -g -G -c -arch=sm_70 $(OPTFLAG) cuallocate.cu -o cuallocate.o
	mpicc -g -Wall -c simutilsmpicuda.c -o simutilsmpicuda.o -lm  	# optimization breaks this!!
	mpicc -g -Wall $(OPTFLAG) delnetstdpinputmpi.o delnetmpi.o spkrcd.o paramutils.o simkernelsmpicuda.o cuallocate.o simutilsmpicuda.o -o runtrial-mpicuda $(LINKFLAGSAIMOS)

aimosnocuda: delnetstdpinputmpi.c delnetmpi.c delnetmpi.h spkrcd.c spkrcd.h paramutils.c paramutils.h simkernelsmpicuda.cu simkernelsmpicuda.h simutilsmpi.c simutilsmpi.h cuallocate.cu cuallocate.h
	mpicc -g -Wall $(OPTFLAG) -c delnetstdpinputmpi.c -o delnetstdpinputmpi.o
	mpicc -g -Wall $(OPTFLAG) -c delnetmpi.c -o delnetmpi.o 
	gcc -g -Wall $(OPTFLAG) -c spkrcd.c -o spkrcd.o	
	gcc -g -Wall $(OPTFLAG) -c paramutils.c -o paramutils.o
	nvcc -g -G -c -arch=sm_70 $(OPTFLAG) simkernelsmpicuda.cu -o simkernelsmpicuda.o -lm
	nvcc -g -G -c -arch=sm_70 $(OPTFLAG) cuallocate.cu -o cuallocate.o
	mpicc -g -Wall -c simutilsmpi.c -o simutilsmpi.o -lm  	# optimization breaks this!!
	mpicc -g -Wall $(OPTFLAG) delnetstdpinputmpi.o delnetmpi.o spkrcd.o paramutils.o simkernelsmpicuda.o cuallocate.o simutilsmpi.o -o runtrial-mpi $(LINKFLAGSAIMOS)

clean:
	rm *.o runtrial-mpi*


