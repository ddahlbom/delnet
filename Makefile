CC= gcc
CFLAGS= -g -O3 -Wall -fopenmp -I./ -L./
LDLIBS = -lm -fopenmp

default: delnetstdp


delnetsketch: delnetsketch.o delnet.o
	$(CC) -o sketch-exec delnet.o delnetsketch.o $(LDLIBS)

delnetstdp: delnetstdp.o delnet.o spkrcd.o
	$(CC) -o stdp-exec delnet.o delnetstdp.o spkrcd.o $(LDLIBS)

cuda: delnetstdpcuda.o delnetcuda.o
	nvcc -o stdpcuda-exec delnetstdpcuda.o delnetcuda.o

delnetstdpcuda.o: delnetstdpcuda.cu delnetcuda.o
	nvcc -g -G -I./ -L./ -c delnetstdpcuda.cu

delnetcuda.o: delnetcuda.cu delnetcuda.h
	nvcc -g -G -c delnetcuda.cu

delnetstdp.o: delnetstdp.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetstdp.c

delnetsketch.o: delnetsketch.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetsketch.c 

spkrcd.o: spkrcd.c spkrcd.h
	$(CC) $(CFLAGS) $(LDLIBS) -c spkrcd.c

delnet.o: delnet.c delnet.h
	$(CC) $(CFLAGS) $(LDLIBS) -c delnet.c

clean:
	rm delnetsketch.o delnet.o sketch-exec stdp-exec delnetstdp.o delnetstdpcuda.o delnetcuda.o stdpcuda-exec
