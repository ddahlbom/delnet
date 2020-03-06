CC= gcc
CFLAGS= -g -Wall -I./ -L./
LDLIBS = -lm

default: delnetsketch

delnetsketch: delnetsketch.o delnet.o
	$(CC) -o sketch-exec delnet.o delnetsketch.o $(LDLIBS)

delnetsketch.o: delnetsketch.c delnet.o
	$(CC) $(CFLAGS) $(LDLIBS) -c delnetsketch.c 


delnet.o: delnet.c delnet.h
	$(CC) $(CFLAGS) $(LDLIBS) -c delnet.c

clean:
	rm delnetsketch.o delnet.o sketch-exec
