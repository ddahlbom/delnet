CC= mpicc 
CFLAGS= -g -Wall 
LDLIBS = -lm 

default: optim
optim: CFLAGS += -O3
optim: main
sanitize: CFLAGS += -fsanitize=address
sanitize: debug
debug: main

main: delnetmain.o delnetmpi.o spkrcd.o paramutils.o simkernelsmpi.o simutilsmpi.o
	$(CC) $(CFLAGS) -o runtrial-mpi delnetmpi.o delnetmain.o spkrcd.o paramutils.o simkernelsmpi.o simutilsmpi.o $(LDLIBS) 

delnetmain.o: delnetmain.c delnetmpi.o
	$(CC) $(CFLAGS) -c delnetmain.c

simkernelsmpi.o: simkernelsmpi.c simkernelsmpi.h
	$(CC) $(CFLAGS) -c simkernelsmpi.c

simutilsmpi.o: simutilsmpi.c simutilsmpi.h
	$(CC) $(CFLAGS) -c simutilsmpi.c

paramutils.o: paramutils.c paramutils.h
	$(CC) $(CFLAGS) -c paramutils.c

spkrcd.o: spkrcd.c spkrcd.h
	$(CC) $(CFLAGS) -c spkrcd.c

delnetmpi.o: delnetmpi.c delnetmpi.h
	$(CC) $(CFLAGS) -c delnetmpi.c

delnet.o: delnet.c delnet.h
	$(CC) $(CFLAGS) $(LDLIBS) -c delnet.c

clean:
	rm *.o runtrial-mpi 
