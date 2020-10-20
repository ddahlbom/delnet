CC= mpicc 
CFLAGS= -g -Wall 
LDLIBS = -lm 

default: optim
optim: CFLAGS += -O3
optim: main
sanitize: CFLAGS += -fsanitize=address
sanitize: pgsearch-exec
debug: main
pg-debug: CFLAGS += -O1
pg-debug: pgsearch-exec
pg: CFLAGS += -O3
pg: pgsearch-exec

pgsearch-exec: pgsearchmain.o pgsearch.o delnet.o spkrcd.o paramutils.o simkernels.o simutils.o inputs.o
	$(CC) $(CFLAGS) -o ../pgsearch-mpi delnet.o spkrcd.o pgsearch.o pgsearchmain.o paramutils.o simkernels.o simutils.o inputs.o $(LDLIBS) 

pgsearchmain.o: pgsearchmain.c 
	$(CC) $(CFLAGS) -c pgsearchmain.c

main: delnetmain.o delnet.o spkrcd.o paramutils.o simkernels.o simutils.o inputs.o
	$(CC) $(CFLAGS) -o ../runtrial-mpi delnet.o delnetmain.o spkrcd.o paramutils.o simkernels.o simutils.o inputs.o $(LDLIBS) 

delnetmain.o: delnetmain.c delnet.o
	$(CC) $(CFLAGS) -c delnetmain.c

pgsearch.o: pgsearch.c pgsearch.h
	$(CC) $(CFLAGS) -c pgsearch.c

inputs.o: inputs.c inputs.h
	$(CC) $(CFLAGS) -c inputs.c

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
