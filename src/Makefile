CC= mpicc 
CFLAGS= -g -Wall 
LDLIBS = -lm 

default: optim
optim: CFLAGS += -O3
optim: main
sanitize: CFLAGS += -fsanitize=address
sanitize: debug
debug: main
pg-debug: pgsearchmain
pg: CFLAGS += -O3
pg: pgsearchmain

main: delnetmain.o delnet.o spkrcd.o paramutils.o simkernels.o simutils.o
	$(CC) $(CFLAGS) -o ../runtrial-mpi delnet.o delnetmain.o spkrcd.o paramutils.o simkernels.o simutils.o $(LDLIBS) 

pgsearchmain: pgsearch.o delnet.o spkrcd.o paramutils.o simkernels.o simutils.o
	$(CC) $(CFLAGS) -o ../pgsearch-mpi delnet.o spkrcd.o pgsearch.o paramutils.o simkernels.o simutils.o $(LDLIBS) 

pgsearch.o: pgsearch.c
	$(CC) $(CFLAGS) -c pgsearch.c

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

delnet.o: delnet.c delnet.h
	$(CC) $(CFLAGS) -c delnet.c -o delnet.o

clean:
	rm *.o
