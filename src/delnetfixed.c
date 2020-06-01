#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>

// #include "delnetfixed.h"
#define DNF_BUF_SIZE 15  // 2^n - 1

#define data_t double
#define idx_t unsigned long
#define mpi_idx_t MPI_UNSIGNED_LONG

typedef enum dnf_error {
	DNF_SUCCESS,
	DNF_BUFFER_OVERFLOW
} dnf_error;

/* --------------------	Structures -------------------- */
typedef struct dnf_node_s {
	idx_t *numtargetranks;
	idx_t *numtargetsperrank;
	idx_t *targets;
	idx_t numinputs;
	idx_t inputoffset;
} dnf_node;


typedef struct dnf_delaybuf_s {
	unsigned short delaylen;
	unsigned short counts[DNF_BUF_SIZE];	
} dnf_delaybuf;


typedef struct dnf_delaynet_s {
	//idx_t numsendranks;
	//idx_t *eventsendbuflens;
	//idx_t *eventsendbufs;
	//idx_t numrecvranks;
	//idx_t *eventrecvbuflens;
	//idx_t *eventrecvbufs;

	//dnf_node *nodes;
	idx_t numnodes;
	data_t *nodeinputbuf;
	idx_t *nodeinputoffsets;
	idx_t *numbuffers; // per node
	idx_t numbufferstotal;
	dnf_delaybuf *buffers;
} dnf_delaynet;


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


/* --------------------	MPI Indexing Utils -------------------- */
typedef struct rankidx_s {
	int commrank;
	idx_t idx;
} rankidx;


idx_t dnf_maxnode(int commrank, int commsize, idx_t numpoints)
{
	idx_t basesize = floor(numpoints/(idx_t)commsize);
	return commrank < (numpoints % commsize) ? basesize + 1 : basesize;
}


idx_t dnf_nodeoffset(int commrank, int commsize, idx_t numpoints)
{
	idx_t offset = 0;
	int i=0;
	while (i < commrank) {
		offset += dnf_maxnode(commrank, commsize, numpoints);
		i++;
	}
	return offset;
}

idx_t *dnf_getstartidcs(int commsize, idx_t numpoints)
{
	idx_t *startidcs = malloc(sizeof(idx_t)*commsize);
	startidcs[0] = 0;
	for (idx_t rank=1; rank<commsize; rank++) {
		startidcs[rank] = dnf_nodeoffset(rank, commsize, numpoints);
	}
	return startidcs;
}

rankidx globaltolocal(idx_t idx_g, idx_t *startidcs)
{
	rankidx idx_l;



	return idx_l;
}


idx_t localtoglobal(rankidx idx_l, idx_t *startidcs)
{
	idx_t idx_g;

	return idx_g;
}


typedef struct dnf_listnode_uint_s { 	
	unsigned long val;
	struct dnf_listnode_uint_s *next;
} dnf_listnode_uint;


typedef struct dnf_list_uint_s {
	unsigned long count;
	struct dnf_listnode_uint_s *head;
} dnf_list_uint;


dnf_list_uint *dnf_list_uint_init() {
	dnf_list_uint *newlist;
	newlist = malloc(sizeof(dnf_list_uint));
	newlist->count = 0;
	newlist->head = NULL; 
	return newlist;
}


void dnf_list_uint_push(dnf_list_uint *l, unsigned long val) {
	dnf_listnode_uint *newnode;
	newnode = malloc(sizeof(dnf_listnode_uint));
	newnode->val = val;
	newnode->next = l->head;
	l->head = newnode;
	l->count += 1;
}


unsigned long dnf_list_uint_pop(dnf_list_uint *l) {
	unsigned long val;
	dnf_listnode_uint *temp;
	if (l->head != NULL) {
		val = l->head->val;
		temp = l->head;
		l->head = l->head->next;
		free(temp);
		l->count -= 1;
	}
	else {
		printf("Tried to pop an empty list. Exiting.\n");
		exit(-1);
	}
	return val;
}

void dnf_list_uint_free(dnf_list_uint *l) {
	while (l->head != NULL) {
		dnf_list_uint_pop(l);
	}
	free(l);
}


/* --------------------	Primary Delaynet Functions -------------------- */

/*
 * Highly un-optimized delnet initialization function
 */
dnf_delaynet *dnf_delaynetfromgraph(unsigned long *graph, unsigned long n,
									int commrank, int commsize)
{
	dnf_delaynet *dn = malloc(sizeof(dnf_delaynet));

	/* Establish node parititioning across cranks */
	idx_t *startidcs;
	if (commrank==0) 
		startidcs = dnf_getstartidcs(commsize, n);
	else 
		startidcs = malloc(sizeof(idx_t)*commsize);
	MPI_Bcast(startidcs, commsize, mpi_idx_t, 0, MPI_COMM_WORLD);


	/* Build process-local delnet infrastructure */
	idx_t n1, n2;
	n1 = startidcs[commrank];
	n2 = commrank < commsize - 1 ? startidcs[commrank+1] : n;
	dn->numnodes = n2-n1;


	/* Count number of inputs for each node (forgive column major indexing) */
	idx_t *numinputs = 0;
	numinputs = calloc(n2-n1, sizeof(idx_t));
	for (idx_t c=n1; c<n2; c++) {
		for (idx_t r=0; r<n; r++) 
			if (graph[r*n+c] != 0) numinputs[c] += 1;
	}
	

	/* initialize delaylines and record their origin */
	idx_t *bufferinputnodes = 0;
	idx_t *bufferoffsets = 0;
	idx_t numinputstotal = 0;

	bufferoffsets = malloc(sizeof(idx_t)*(n2-n1));
	for (idx_t i=0; i<n2-n1; i++) {
		bufferoffsets[i] = numinputstotal;
		numinputstotal += numinputs[i];
	}

	dn->buffers = malloc(sizeof(dnf_delaybuf)*numinputstotal);
	dn->nodeinputbuf = calloc(numinputstotal, sizeof(data_t));
	dn->numbuffers = numinputs;
	dn->numbufferstotal = numinputstotal;
	dn->nodeinputoffsets = bufferoffsets;

	bufferinputnodes = malloc(sizeof(idx_t)*numinputstotal);
	idx_t counter = 0;
	idx_t counter_old = 0;
	for (idx_t c=n1; c<n2; c++) {
		for (idx_t r=0; r<n; r++) {
			if(graph[r*n+c] != 0) {
				dnf_bufinit(&dn->buffers[counter], graph[r*n+c]);
				bufferinputnodes[counter] = r;
				counter++;
			}
		}
	}

	return dn;
}





/* -------------------- Testing -------------------- */
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

int main(int argc, char *argv[]) 
{

	/* Init MPI */
	int commsize, commrank;
	MPI_Init(&argc, &argv);	
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &commrank);

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

	/* test partitioning */
	idx_t numpoints = 1000;
	idx_t numranks = 7;
	idx_t *startidcs = dnf_getstartidcs(numranks, numpoints);
	for (int i=0; i<numranks; i++)
		printf("Start index on rank %d: %lu\n", i, startidcs[i]);


	/* test delnet from graph */
	unsigned long graph[16] = { [0] = 0, [1] = 2, [2] = 5, [3] = 0,
								[4] = 0, [5] = 0, [6] = 3, [7] = 3,
								[8] = 0, [9] = 0, [10]= 4, [11]= 4,
								[12]= 5, [13]= 0, [14]= 0, [15]= 0 };

	dnf_delaynet *dn = dnf_delaynetfromgraph(graph, 4, commrank, commsize);

	return 0;
}
