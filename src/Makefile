CC= mpicc 
CFLAGS= -g -Wall 
LDLIBS = -lm 

default: optim
optim: CFLAGS += -O3
optim: main
sanitize: CFLAGS += -fsanitize=address
sanitize: debug
debug: main

main: delnetmain.o delnet.o spkrcd.o paramutils.o simkernels.o simutils.o
	$(CC) $(CFLAGS) -o ../runtrial-mpi delnet.o delnetmain.o spkrcd.o paramutils.o simkernels.o simutils.o $(LDLIBS) 

delnetmain.o: delnetmain.c delnet.o
	$(CC) $(CFLAGS) -c delnetmain.c

simkernels.o: simkernels.c simkernels.h
	$(CC) $(CFLAGS) -c simkernels.c

simutils.o: simutils.c simutils.h
	$(CC) $(CFLAGS) -c simutils.c

paramutils.o: paramutils.c paramutils.h
	$(CC) $(CFLAGS) -c paramutils.c

spkrcd.o: spkrcd.c spkrcd.h
	$(CC) $(CFLAGS) -c spkrcd.c

delnet.o: delnetfixed.c delnetfixed.h
	$(CC) $(CFLAGS) -c delnetfixed.c -o delnet.o

clean:
	rm *.o ../runtrial-mpi 
