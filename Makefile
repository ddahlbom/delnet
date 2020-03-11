CC= gcc
CFLAGS= -g -O3 -Wall -I./ -L./
LDLIBS = -lm

default: delnetstdp


delnetsketch: delnetsketch.o delnet.o
	$(CC) -o sketch-exec delnet.o delnetsketch.o $(LDLIBS)

delnetstdp: delnetstdp.o delnet.o
	$(CC) -o stdp-exec delnet.o delnetstdp.o $(LDLIBS)

delnetcuda: delnetstdpcuda.o delnet.o
	$(CC) -o delnetcuda-exec delnet.o delnetstdpcuda.o -lcudart

delnetcuda.o: delnetstdpcuda.cu delnet.o
	nvcc -g -G -Xcompiler -Wall -c delnetstdpcuda.cu

delnetstdp.o: delnetstdp.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetstdp.c

delnetsketch.o: delnetsketch.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetsketch.c 


delnet.o: delnet.c delnet.h
	$(CC) $(CFLAGS) $(LDLIBS) -c delnet.c

clean:
	rm delnetsketch.o delnet.o sketch-exec stdp-exec delnetstdp.o
