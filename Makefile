CC= gcc
CFLAGS= -g -Wall -I./ -L./
LDLIBS = -lm 

default: mpioptim
debug: mpidebug

mpidebug: delnetmain.o delnetmpi.o spkrcd.o paramutils.o simkernelsmpi.o simutilsmpi.o
	mpicc -g -o runtrial-mpi delnetmpi.o delnetmain.o spkrcd.o paramutils.o simkernelsmpi.o simutilsmpi.o $(LDLIBS)

mpioptim: delnetmain.o delnetmpi-opt.o spkrcd.o paramutils.o simkernelsmpi-opt.o simutilsmpi-opt.o
	mpicc -g -O3 -o runtrial-mpi delnetmpi-opt.o delnetmain.o spkrcd.o paramutils.o simkernelsmpi-opt.o simutilsmpi-opt.o $(LDLIBS)

delnetmain-opt.o: delnetmain.c delnetmpi.o
	mpicc $(CFLAGS) -O3 $(LDLIBS) -c delnetmain.c -o delnetmain-opt.o

delnetmain.o: delnetmain.c delnetmpi.o
	mpicc $(CFLAGS) $(LDLIBS) -c delnetmain.c

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
	mpicc $(CFLAGS) -O3 $(LDLIBS) -o simkernelsmpi-opt.o -c simkernelsmpi.c

simkernelsmpi.o: simkernelsmpi.c simkernelsmpi.h
	mpicc $(CFLAGS) $(LDLIBS) -c simkernelsmpi.c

simkernels-opt.o: simkernels.c simkernels.h
	$(CC) $(CFLAGS) -O3 $(LDLIBS) -o simkernels-opt.o -c simkernels.c

simkernels.o: simkernels.c simkernels.h
	$(CC) $(CFLAGS) $(LDLIBS) -c simkernels.c

simutilsmpi-opt.o: simutilsmpi.c simutilsmpi.h
	mpicc $(CFLAGS) -O3 $(LDLIBS) -o simutilsmpi-opt.o -c simutilsmpi.c

simutilsmpi.o: simutilsmpi.c simutilsmpi.h
	mpicc $(CFLAGS) $(LDLIBS) -c simutilsmpi.c

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
	rm *.o runtrial-mpi 
