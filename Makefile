CFLAGS= -g -Wall
LDLIBS = -lm

all: delnetsketch.c
	gcc $(CFLAGS) $< -o sketchexec $(LDLIBS)
