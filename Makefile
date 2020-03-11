CC= gcc
CFLAGS= -g -O3 -Wall -I./ -L./
LDLIBS = -lm

default: delnetstdp


delnetsketch: delnetsketch.o delnet.o
	$(CC) -o sketch-exec delnet.o delnetsketch.o $(LDLIBS)

delnetstdp: delnetstdp.o delnet.o
	$(CC) -o stdp-exec delnet.o delnetstdp.o $(LDLIBS)

cuda: delnetstdpcuda.cu
	nvcc -g -G -O3 -o stdpcuda-exe delnetstdpcuda.cu

delnetstdp.o: delnetstdp.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetstdp.c

delnetsketch.o: delnetsketch.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetsketch.c 


delnet.o: delnet.c delnet.h
	$(CC) $(CFLAGS) $(LDLIBS) -c delnet.c

clean:
	rm delnetsketch.o delnet.o sketch-exec stdp-exec delnetstdp.o
