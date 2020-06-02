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
	//dnf_node *nodes;
	idx_t numnodes;
	data_t *nodeinputbuf;
	idx_t *nodebufferoffsets;
	idx_t *numbuffers; // per node
	idx_t numbufferstotal;
	dnf_delaybuf *buffers;

	idx_t **dests; 	//destination indexed [target rank][local neuron number]
	idx_t **destoffsets;
	idx_t **destlens;
	idx_t *destlenstot; // maximum number of targets per rank
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

idx_t *dnf_getlens(int commsize, idx_t numpoints)
{
	idx_t *lens = malloc(sizeof(idx_t)*commsize);
	for (idx_t i=0; i<commsize; i++)
		lens[i] = dnf_maxnode(i, commsize, numpoints);
	return lens;
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



typedef struct dnf_listnode_uint_s { 	
	unsigned long val;
	struct dnf_listnode_uint_s *next;
} dnf_listnode_uint;


typedef struct dnf_idxlist_s {
	unsigned long count;
	struct dnf_listnode_uint_s *head;
} dnf_idxlist;


dnf_idxlist *dnf_idxlist_init() {
	dnf_idxlist *newlist;
	newlist = malloc(sizeof(dnf_idxlist));
	newlist->count = 0;
	newlist->head = NULL; 
	return newlist;
}


void dnf_idxlist_push(dnf_idxlist *l, unsigned long val) {
	dnf_listnode_uint *newnode;
	newnode = malloc(sizeof(dnf_listnode_uint));
	newnode->val = val;
	newnode->next = l->head;
	l->head = newnode;
	l->count += 1;
}


unsigned long dnf_idxlist_pop(dnf_idxlist *l) {
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

void dnf_idxlist_free(dnf_idxlist *l) {
	while (l->head != NULL) {
		dnf_idxlist_pop(l);
	}
	free(l);
}


/* --------------------	Primary Delaynet Functions -------------------- */

int cmp (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void dfn_synctargetinfo(idx_t **destoffsets, idx_t **dests,
					    idx_t **destlens, idx_t *destlenstot,
					    dnf_delaynet *dn, idx_t *nodesperrank,
					    int commsize)
{

	idx_t **destoffsets_l = malloc(sizeof(idx_t*)*commsize);
	idx_t **destlens_l = malloc(sizeof(idx_t*)*commsize);
	idx_t **dests_l = malloc(sizeof(idx_t*)*commsize);
	idx_t *destlenstot_l = malloc(sizeof(idx_t)*commsize);

	for (idx_t r=0; r<commsize; r++) {
		destoffsets_l[r] = malloc(sizeof(idx_t)*nodesperrank[r]);
		destlens_l[r] = malloc(sizeof(idx_t)*nodesperrank[r]);
	}

	MPI_Request *sendreqoffsets = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *sendreqlens = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *sendreqlentot = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *sendreqdests = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *recvreqoffsets = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *recvreqlens = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *recvreqlentot = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *recvreqdests = malloc(sizeof(MPI_Request)*commsize);

	/* Send node destinations metadata to this rank to all other ranks */
	for (idx_t r=0; r<commsize; r++) {
		MPI_Isend(destoffsets[r],
				  nodesperrank[r],
				  mpi_idx_t,
				  r,
				  0,
				  MPI_COMM_WORLD,
				  &sendreqoffsets[r]);
		MPI_Isend(destlens[r],
				  nodesperrank[r],
				  mpi_idx_t,
				  r,
				  1,
				  MPI_COMM_WORLD,
				  &sendreqlens[r]);
		MPI_Isend(&destlenstot[r],
				  1,
				  mpi_idx_t,
				  r,
				  2,
				  MPI_COMM_WORLD,
				  &sendreqlentot[r]);
	}

	/* Recieve destinations metadata from other processes */
	for (idx_t r=0; r<commsize; r++) {
		MPI_Irecv(destoffsets_l[r],
				  nodesperrank[r],
				  mpi_idx_t,
				  r,
				  0,
				  MPI_COMM_WORLD,
				  &recvreqoffsets[r]);
		MPI_Irecv(destlens_l[r],
				  nodesperrank[r],
				  mpi_idx_t,
				  r,
				  1,
				  MPI_COMM_WORLD,
				  &recvreqlens[r]);
		MPI_Irecv(&destlenstot_l[r],
				  1,
				  mpi_idx_t,
				  r,
				  2,
				  MPI_COMM_WORLD,
				  &recvreqlentot[r]);
	}

	/* Wait for transactions to complete */
	for (idx_t r=0; r<commsize; r++) {
		MPI_Wait(&recvreqoffsets[r], MPI_STATUS_IGNORE);
		MPI_Wait(&recvreqlens[r], MPI_STATUS_IGNORE);
		MPI_Wait(&recvreqlentot[r], MPI_STATUS_IGNORE);
		MPI_Wait(&sendreqoffsets[r], MPI_STATUS_IGNORE);
		MPI_Wait(&sendreqlens[r], MPI_STATUS_IGNORE);
		MPI_Wait(&sendreqlentot[r], MPI_STATUS_IGNORE);
	}

	for (idx_t r=0; r<commsize; r++) 
		dests_l[r] = malloc(sizeof(idx_t)*destlenstot_l[r]);

	/* Send node destinations to this rank to all other ranks */
	for (idx_t r=0; r<commsize; r++) {
		MPI_Isend(dests[r],
				  destlenstot[r],
				  mpi_idx_t,
				  r,
				  3,
				  MPI_COMM_WORLD,
				  &sendreqdests[r]);
	}

	/* Recieve destinations from other processes */
	for (idx_t r=0; r<commsize; r++) {
		MPI_Irecv(dests_l[r],
				  destlenstot_l[r],
				  mpi_idx_t,
				  r,
				  3,
				  MPI_COMM_WORLD,
				  &recvreqdests[r]);
	}

	for (idx_t r=0; r<commsize; r++) {
		MPI_Wait(&recvreqdests[r], MPI_STATUS_IGNORE);
		MPI_Wait(&sendreqdests[r], MPI_STATUS_IGNORE);
	}

	free(sendreqoffsets);
	free(sendreqlens);
	free(sendreqlentot);
	free(sendreqdests);
	free(recvreqoffsets);
	free(recvreqlens);
	free(recvreqlentot);
	free(recvreqdests);

	/* Sort the destinations locally */
	for (idx_t r=0; r<commsize; r++) 
		for (idx_t n=0; n<nodesperrank[r]; n++) 
			qsort(&dests_l[r][destoffsets_l[r][n]], destlens_l[r][n],
					sizeof(idx_t), cmp);

	dn->destoffsets = destoffsets_l;
	dn->destlens = destlens_l;
	dn->dests = dests_l;
	dn->destlenstot = destlenstot_l;
}


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
	idx_t *nodesperrank = dnf_getlens(commsize, n);
	n1 = startidcs[commrank];
	n2 = commrank < commsize - 1 ? startidcs[commrank+1] : n;
	dn->numnodes = n2-n1;


	/* Count number of inputs for each node (forgive column major indexing) */
	idx_t *numinputs = 0;
	numinputs = calloc(n2-n1, sizeof(idx_t));
	for (idx_t c=n1; c<n2; c++) {
		for (idx_t r=0; r<n; r++) 
			if (graph[r*n+c] != 0) numinputs[c-n1] += 1;
	}
	

	/* ----- Initialize delaylines and record their origin ----- */
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
	dn->nodebufferoffsets = bufferoffsets;

	bufferinputnodes = malloc(sizeof(idx_t)*numinputstotal);
	idx_t counter = 0;
	for (idx_t c=n1; c<n2; c++) {
		for (idx_t r=0; r<n; r++) {
			if(graph[r*n+c] != 0) {
				dnf_bufinit(&dn->buffers[counter], graph[r*n+c]);
				bufferinputnodes[counter] = r; 	// ***
				counter++;
			}
		}
	}

	// ASSERTION -- Delete later -- just for logic testing
	if (counter != numinputstotal) {
		printf("counter: %lu, total inputs: %lu\n", counter, numinputstotal);
		exit(-1);
	}


	/* ---------- global node to process-local target bookkeeping ---------- */

	/* Initialize lists of output destinations for each node */
	dnf_idxlist **nodedestinations = malloc(sizeof(dnf_idxlist *)*n); 
	for (idx_t i=0; i<n; i++)
		nodedestinations[i] = dnf_idxlist_init();

	/* populate output destinations */
	for (idx_t i=0; i<numinputstotal; i++)
		dnf_idxlist_push(nodedestinations[bufferinputnodes[i]], i);

	
	/* consolidate output destinations into multiple arrays */
	idx_t **destoffsets = malloc(sizeof(idx_t*)*commsize);
	idx_t **destlens = malloc(sizeof(idx_t*)*commsize);
	idx_t ***destslists = malloc(sizeof(idx_t**)*commsize);
	idx_t *destlenstot = malloc(sizeof(idx_t)*commsize);

	for (idx_t sd=0; sd<commsize; sd++) {
		idx_t i1, i2, numnodes_l;
		i1 = startidcs[sd];
		i2 = sd < commsize - 1 ? startidcs[sd+1] : n;
		numnodes_l = i2-i1;
		if (numnodes_l != nodesperrank[sd]) {printf("Nooo...\n"); exit(-1);}
		destslists[sd] = malloc(sizeof(idx_t*)*numnodes_l);
		destlens[sd] = malloc(sizeof(idx_t)*numnodes_l);
		destoffsets[sd] = malloc(sizeof(idx_t)*numnodes_l);
		idx_t runningoffset = 0;
		for (idx_t i=0; i<i2-i1; i++) {
			destoffsets[sd][i] = runningoffset;
			destlens[sd][i] = nodedestinations[i+i1]->count;
			destslists[sd][i] = malloc(sizeof(idx_t)*destlens[sd][i]);
			for (idx_t j=0; j<destlens[sd][i]; j++)
				/* note - i1 -- so in process local indexing */
				destslists[sd][i][j] = dnf_idxlist_pop(nodedestinations[i+i1]); 
			dnf_idxlist_free(nodedestinations[i+i1]);
			runningoffset += destlens[sd][i];
		}
		destlenstot[sd] = runningoffset;
	}
	free(nodedestinations);

	/* Consolidate list of destinations into master array */
	idx_t **dests = malloc(sizeof(idx_t*)*commsize);
	for (idx_t sd = 0; sd<commsize; sd++) {
		dests[sd] = malloc(sizeof(idx_t)*destlenstot[sd]);
		idx_t i1, i2, numnodes_l;
		i1 = startidcs[sd];
		i2 = sd < commsize-1 ? startidcs[sd+1] : n;
		numnodes_l = i2-i1;
		idx_t c = 0;
		for (idx_t i=i1; i<i2; i++) {
			for (idx_t j=0; j<destlens[sd][i-i1]; j++) {
				dests[sd][c] = destslists[sd][i-i1][j];
				c++;
			}
			if (destlens[sd][i-i1] > 0)
				free(destslists[sd][i-i1]);
		}
		//ASSERTION -- logic test only
		if (c != destlenstot[sd]) {
			printf("Bad transfer of desintation array\n");
			printf("c: %lu, destlenstot: %lu\n", c, destlenstot[sd]);
			exit(-1);
		}
		free(destslists[sd]);
	}
	free(destslists);


	/* Set up data for receiving target info */
	dfn_synctargetinfo(destoffsets, dests, destlens, destlenstot,
					   dn, nodesperrank, commsize);


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

	/*
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

	// test partitioning 
	idx_t numpoints = 1000;
	idx_t numranks = 7;
	idx_t *startidcs = dnf_getstartidcs(numranks, numpoints);
	for (int i=0; i<numranks; i++)
		printf("Start index on rank %d: %lu\n", i, startidcs[i]);
	free(startidcs);
	*/


	/* test delnet from graph */
	unsigned long graph[16] = { [0] = 0, [1] = 2, [2] = 5, [3] = 0,
								[4] = 0, [5] = 0, [6] = 3, [7] = 3,
								[8] = 0, [9] = 0, [10]= 0, [11]= 4,
								[12]= 5, [13]= 0, [14]= 0, [15]= 0 };

	dnf_delaynet *dn = dnf_delaynetfromgraph(graph, 4, commrank, commsize);

	MPI_Finalize();

	return 0;
}
