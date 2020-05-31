#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>

// #include "delnetfixed.h"
#define DNF_BUF_SIZE 15  // 2^n - 1

#define data_t double
#define idx_t unsigned long

typedef enum dnf_error {
	DNF_SUCCESS,
	DNF_BUFFER_OVERFLOW
} dnf_error;

/* --------------------	Structures -------------------- */
typedef struct dnf_delaynet_s {
	idx_t numsendranks;
	idx_t *eventsendbuflens;
	idx_t *eventsendbufs;
	idx_t numrecvranks;
	idx_t *eventrecvbuflens;
	idx_t *eventrecvbufs;
} dnf_delaynet;


typedef struct dnf_node_s {
	idx_t *numtargetranks;
	idx_t *numtargetsperrank;
	idx_t *targets;
} dnf_node;


typedef struct dnf_delaybuf_s {
	unsigned short delaylen;
	unsigned short counts[DNF_BUF_SIZE];	
} dnf_delaybuf;


/* --------------- Buffer Functions  ---------------*/
static inline dnf_error dnf_bufinit(dnf_delaybuf *buf, unsigned short len)
{
	buf->delaylen = len;
	for (idx_t i=0; i<DNF_BUF_SIZE; i++) 
		buf->counts[i] = 0;
	return DNF_SUCCESS;
}

static inline dnf_error dnf_recordevent(dnf_delaybuf *buf)
{
	bool eventrecorded = false;
	idx_t i = 0;
	while (i < DNF_BUF_SIZE && !eventrecorded) {
		if (buf->counts[i] == 0) {
			buf->counts[i] = buf->delaylen;
			eventrecorded = true;
		}
		i++;
	}
	return eventrecorded ? DNF_SUCCESS : DNF_BUFFER_OVERFLOW;
}

/* Cycles through all possible stored events -- OPTIMIZE LATER */
static inline dnf_error dnf_bufadvance(dnf_delaybuf *buf, data_t *out)
{
	*out = 0.0;
	for (idx_t i=0; i<DNF_BUF_SIZE; i++) {
		if (buf->counts[i] > 1) {
			buf->counts[i] -= 1;
		} else if (buf->counts[i] == 1) {
			buf->counts[i] = 0;
			*out = 1.0;
		}
	}
	return DNF_SUCCESS;
}


/* --------------------	MPI Utils -------------------- */
typedef struct rankidx_s {
	int commrank;
	idx_t idx;
} rankidx;

idx_t dnf_maxnode(int rank, int commsize, idx_t numpoints)
{
	idx_t basesize = floor(numpoints/(idx_t)commsize);
	return rank < (numpoints % commsize) ? basesize + 1 : basesize;
}

idx_t dnf_nodeoffset(int rank, int commsize, idx_t numpoints)
{
	idx_t offset = 0;
	int i=0;
	while (i < rank) {
		offset += dnf_maxnode(rank, commsize, numpoints);
		i++;
	}
	return offset;
}


/* --------------------	Primary Delaynet Functions -------------------- */
dnf_delaynet *dnf_delaynetfromgraph(unsigned long *graph, unsigned long n,
									int commsize, int commrank)
{
		
}


// for testing only)
bool in(idx_t val, idx_t *vals, idx_t n)
{
	idx_t i = 0;
	bool found = false;
	while (i < n && !found) {
		if (val == vals[i]) found = true;
		i++;
	}
	return found;
}

void main() 
{
	dnf_delaybuf buf;
	dnf_bufinit(&buf, 6);
	data_t output = 0.0;

	idx_t eventtimes[] = {3, 11, 18, 54, 76, 92};
	idx_t n = 6;

	for (idx_t i=0; i<100; i++) {
		printf("Step %lu: %lf\n", i, output);
		if (in(i, eventtimes, n))
			dnf_recordevent(&buf);
		dnf_bufadvance(&buf, &output);
	}
}
