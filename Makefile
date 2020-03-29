CC= gcc
CFLAGS= -g -Wall -I./ -L./
LDLIBS = -lm 

default: stdpoptim


sketch: delnetsketch.o delnet.o
	$(CC) -o sketch-exec delnet.o delnetsketch.o $(LDLIBS)

debug: delnetstdp.o delnet.o spkrcd.o paramutils.o simutils.o
	$(CC) -o stdp-exec delnet.o delnetstdp.o spkrcd.o paramutils.o simutils.o $(LDLIBS)

stdpoptim: delnetstdp-opt.o delnet-opt.o spkrcd.o paramutils.o simutils-opt.o
	$(CC) -o stdp-exec delnet-opt.o delnetstdp-opt.o spkrcd.o paramutils.o simutils-opt.o $(LDLIBS)

cuda: delnetstdpcuda.o delnetcuda.o
	nvcc -o stdpcuda-exec delnetstdpcuda.o delnetcuda.o

delnetstdpcuda.o: delnetstdpcuda.cu delnetcuda.o
	nvcc -g -G -I./ -L./ -c delnetstdpcuda.cu

delnetcuda.o: delnetcuda.cu delnetcuda.h
	nvcc -g -G -c delnetcuda.cu

delnetstdp.o: delnetstdp.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetstdp.c

delnetstdp-opt.o: delnetstdp.c delnet.o
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o delnetstdp-opt.o -c delnetstdp.c

delnetsketch.o: delnetsketch.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetsketch.c 

simutils-opt.o: simutils.c simutils.h
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o simutils-opt.o -c simutils.c

simutils.o: simutils.c simutils.h
	$(CC) $(CFLAGS) $(LDLIBS) -c simutils.c

paramutils.o: paramutils.c paramutils.h
	$(CC) $(CFLAGS) $(LDLIBS) -c paramutils.c

spkrcd.o: spkrcd.c spkrcd.h
	$(CC) $(CFLAGS) $(LDLIBS) -c spkrcd.c

delnet-opt.o: delnet.c delnet.h
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o delnet-opt.o -c delnet.c

delnet.o: delnet.c delnet.h
	$(CC) $(CFLAGS) $(LDLIBS) -c delnet.c

clean:
	rm delnetsketch.o delnet.o sketch-exec stdp-exec delnetstdp.o delnetstdpcuda.o delnetcuda.o stdpcuda-exec
