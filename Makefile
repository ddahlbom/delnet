CC= gcc
CFLAGS= -g -Wall -I./ -L./
LDLIBS = -lm 

default: inputoptim

mpidebug: delnetstdpinputmpi.o delnetmpi.o spkrcd.o paramutils.o simkernels.o simutilsmpi.o
	mpicc -g -o runtrial-mpi delnetmpi.o delnetstdpinputmpi.o spkrcd.o paramutils.o simkernelsmpi.o simutilsmpi.o $(LDLIBS)

mpioptim: delnetstdpinputmpi.o delnetmpi-opt.o spkrcd.o paramutils.o simkernelsmpi-opt.o simutilsmpi-opt.o
	mpicc -g -O3 -o runtrial-mpi delnetmpi-opt.o delnetstdpinputmpi.o spkrcd.o paramutils.o simkernelsmpi-opt.o simutilsmpi-opt.o $(LDLIBS)

inputdebug: delnetstdpinput.o delnet.o spkrcd.o paramutils.o simkernels.o simutils.o
	$(CC) -o runtrial-exec delnet.o delnetstdpinput.o spkrcd.o paramutils.o simkernels.o simutils.o $(LDLIBS)

inputoptim: delnetstdpinput-opt.o delnet-opt.o spkrcd.o paramutils.o simkernels-opt.o simutils-opt.o 
	$(CC) -o runtrial-exec delnet-opt.o delnetstdpinput-opt.o spkrcd.o paramutils.o simkernels-opt.o simutils-opt.o $(LDLIBS)

delnetstdpinputmpi-opt.o: delnetstdpinputmpi.c delnetmpi.o
	mpicc $(CFLAGS) -O3 $(LDLIBS) -c delnetstdpinputmpi.c -o delnetstdpinputmpi-opt.o

delnetstdpinputmpi.o: delnetstdpinputmpi.c delnetmpi.o
	mpicc $(CFLAGS) $(LDLIBS) -c delnetstdpinputmpi.c

delnetstdpinput.o: delnetstdpinput.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetstdpinput.c

delnetstdpinput-opt.o: delnetstdpinput.c delnet.o
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o delnetstdpinput-opt.o -c delnetstdpinput.c

delnetstdp.o: delnetstdp.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetstdp.c

delnetstdp-opt.o: delnetstdp.c delnet.o
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o delnetstdp-opt.o -c delnetstdp.c

delnetsketch.o: delnetsketch.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetsketch.c 

simkernelsmpi-opt.o: simkernelsmpi.c simkernelsmpi.h
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o simkernelsmpi-opt.o -c simkernelsmpi.c

simkernelsmpi.o: simkernelsmpi.c simkernelsmpi.h
	$(CC) $(CFLAGS) $(LDLIBS) -c simkernelsmpi.c

simkernels-opt.o: simkernels.c simkernels.h
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o simkernels-opt.o -c simkernels.c

simkernels.o: simkernels.c simkernels.h
	$(CC) $(CFLAGS) $(LDLIBS) -c simkernels.c

simutilsmpi-opt.o: simutilsmpi.c simutilsmpi.h
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o simutilsmpi-opt.o -c simutilsmpi.c

simutilsmpi.o: simutilsmpi.c simutilsmpi.h
	$(CC) $(CFLAGS) $(LDLIBS) -c simutilsmpi.c

simutils-opt.o: simutils.c simutils.h
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o simutils-opt.o -c simutils.c

simutils.o: simutils.c simutils.h
	$(CC) $(CFLAGS) $(LDLIBS) -c simutils.c

paramutils.o: paramutils.c paramutils.h
	$(CC) $(CFLAGS) $(LDLIBS) -c paramutils.c

spkrcd.o: spkrcd.c spkrcd.h
	$(CC) $(CFLAGS) $(LDLIBS) -c spkrcd.c

delnetmpi-opt.o: delnetmpi.c delnetmpi.h
	mpicc $(CFLAGS) -O3 $(LDLIBS) -c delnetmpi.c -o delnetmpi-opt.o

delnetmpi.o: delnetmpi.c delnetmpi.h
	mpicc $(CFLAGS) $(LDLIBS) -c delnetmpi.c

delnet-opt.o: delnet.c delnet.h
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o delnet-opt.o -c delnet.c

delnet.o: delnet.c delnet.h
	$(CC) $(CFLAGS) $(LDLIBS) -c delnet.c

clean:
	rm *.o stdpinput-exec
