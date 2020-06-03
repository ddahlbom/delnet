#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>

#include "delnetfixed.h"



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
idx_t dnf_maxnode(int commrank, int commsize, idx_t numpoints)
{
	idx_t basesize = floor(numpoints/(idx_t)commsize);
	return commrank < (numpoints % commsize) ? basesize + 1 : basesize;
}


idx_t dnf_nodeoffset(int commrank, int commsize, idx_t numpoints)
{
	idx_t offset = 0;
	for (int i=0; i<commrank; i++)
		offset += dnf_maxnode(i, commsize, numpoints);
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
void dnf_pushevents(dnf_delaynet *dn, idx_t *eventnodes, idx_t numevents,
					int commsize)
{
	/* load send blocks */
	idx_t *counts = calloc(commsize, sizeof(idx_t)); // remove this allocation
	for (idx_t n=0; n<numevents; n++) {
		for (idx_t r=0; r<commsize; r++) {
			counts[r] += dn->destlens[r][eventnodes[n]];
			for (idx_t i=0; i<dn->destlens[r][eventnodes[n]]; i++) {
				dn->sendblocks[r][counts[r]+i] = dn->dests[r][eventnodes[n]];
			}
		}
	}

	// remove these allocations
	MPI_Request *sr_counts = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *sr = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *rr_counts = malloc(sizeof(MPI_Request)*commsize);
	MPI_Request *rr = malloc(sizeof(MPI_Request)*commsize);

	/* non-blocking send */
	for (idx_t r=0; r<commsize; r++) {
		MPI_Isend(&counts[r], 1, mpi_idx_t, r, 0, MPI_COMM_WORLD, &sr_counts[r]);
		MPI_Isend(dn->sendblocks[r], counts[r], mpi_idx_t, r, 1, MPI_COMM_WORLD,
				  &sr[r]);
	}

	/* Receives */
	for (idx_t r=0; r<commsize; r++) 
		MPI_Irecv(&counts[r], 1, mpi_idx_t, r, 0, MPI_COMM_WORLD, &rr_counts[r]);

	for (idx_t r=0; r<commsize; r++) 
		MPI_Wait(&rr_counts[r], MPI_STATUS_IGNORE);

	for (idx_t r=0; r<commsize; r++)
		MPI_Irecv(&dn->recvblocks[r], counts[r], mpi_idx_t, r, 1, MPI_COMM_WORLD,
				  &rr[r]);

	for (idx_t r=0; r<commsize; r++) 
		MPI_Wait(&rr[r], MPI_STATUS_IGNORE);

	/* record events */
	for (idx_t r=0; r<commsize; r++) {
		for (idx_t n=0; n<counts[r]; n++)
			dnf_recordevent(&dn->buffers[dn->recvblocks[r][n]]);
	}

	/* wait for sends to finish */
	for (idx_t r=0; r<commsize; r++) {
		MPI_Wait(&sr[r], MPI_STATUS_IGNORE);
		MPI_Wait(&sr_counts[r], MPI_STATUS_IGNORE);
	}
}


void dnf_advance(dnf_delaynet *dn)
{
	for (idx_t i=0; i<dn->numbufferstotal; i++)
		dnf_bufadvance(&dn->buffers[i], &dn->nodeinputbuf[i]);
}


data_t *dnf_getinputaddress(dnf_delaynet *dn, idx_t node)
{
	return &dn->nodeinputbuf[dn->nodebufferoffsets[node]];
}



int cmp(const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void dfn_synctargetinfo(idx_t **destoffsets, idx_t **dests,
					    idx_t **destlens, idx_t *destlenstot,
					    dnf_delaynet *dn, idx_t *nodesperrank,
					    int commsize)
{
	dn->destoffsets = malloc(sizeof(idx_t*)*commsize);
	dn->destlens = malloc(sizeof(idx_t*)*commsize);
	dn->dests = malloc(sizeof(idx_t*)*commsize);
	dn->destlenstot = malloc(sizeof(idx_t)*commsize);

	for (idx_t r=0; r<commsize; r++) {
		dn->destoffsets[r] = malloc(sizeof(idx_t)*nodesperrank[r]);
		dn->destlens[r] = malloc(sizeof(idx_t)*nodesperrank[r]);
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
		MPI_Irecv(dn->destoffsets[r],
				  nodesperrank[r],
				  mpi_idx_t,
				  r,
				  0,
				  MPI_COMM_WORLD,
				  &recvreqoffsets[r]);
		MPI_Irecv(dn->destlens[r],
				  nodesperrank[r],
				  mpi_idx_t,
				  r,
				  1,
				  MPI_COMM_WORLD,
				  &recvreqlens[r]);
		MPI_Irecv(&dn->destlenstot[r],
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
	}

	for (idx_t r=0; r<commsize; r++) {
		MPI_Wait(&sendreqoffsets[r], MPI_STATUS_IGNORE);
		MPI_Wait(&sendreqlens[r], MPI_STATUS_IGNORE);
		MPI_Wait(&sendreqlentot[r], MPI_STATUS_IGNORE);
	}

	for (idx_t r=0; r<commsize; r++) 
		dn->dests[r] = malloc(sizeof(idx_t)*dn->destlenstot[r]);

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
		MPI_Irecv(dn->dests[r],
				  dn->destlenstot[r],
				  mpi_idx_t,
				  r,
				  3,
				  MPI_COMM_WORLD,
				  &recvreqdests[r]);
	}

	for (idx_t r=0; r<commsize; r++)
		MPI_Wait(&recvreqdests[r], MPI_STATUS_IGNORE);
	for (idx_t r=0; r<commsize; r++)
		MPI_Wait(&sendreqdests[r], MPI_STATUS_IGNORE);

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
			qsort(&dn->dests[r][dn->destoffsets[r][n]], dn->destlens[r][n],
					sizeof(idx_t), cmp);

	/* allocate send and receive blocks */
	dn->sendblocks = malloc(sizeof(idx_t *)*commsize);
	dn->recvblocks = malloc(sizeof(idx_t *)*commsize);

	for (idx_t r=0; r<commsize; r++) {
		dn->sendblocks[r] = malloc(sizeof(idx_t)*dn->destlenstot[r]);
		dn->recvblocks[r] = malloc(sizeof(idx_t)*destlenstot[r]);
	}

}


/*
 * Un-optimized delnet initialization function
 */
dnf_delaynet *dnf_delaynetfromgraph(unsigned long *graph, unsigned long n,
									int commrank, int commsize)
{
	dnf_delaynet *dn = malloc(sizeof(dnf_delaynet));

	if (n < commsize) {
		printf("Must have fewer processes than nodes!\n Exiting.\n");
		exit(-1);
	}

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


	/* Allocate local send/recv blocks */
	// Already have send size information
	dn->sendblocks = malloc(sizeof(idx_t *)*commsize);
	for (idx_t r=0; r<commsize; r++)
		dn->sendblocks[r] = malloc(sizeof(idx_t)*destlenstot[r]);

	// Gather receive size information


	/* Clean up remaining unused allocations */
	free(startidcs);
	free(nodesperrank);
	free(bufferinputnodes);
	free(destlenstot);
	for (idx_t r=0; r<commsize; r++) {
		free(destoffsets[r]);
		free(destlens[r]);
		free(dests[r]);
	}
	free(dests);
	free(destoffsets);
	free(destlens);

	return dn;
}


void dnf_freedelaynet(dnf_delaynet *dn, int commsize)
{
	free(dn->nodeinputbuf);
	free(dn->nodebufferoffsets);
	free(dn->numbuffers);
	free(dn->buffers);

	for (idx_t i=0; i<commsize; i++) {
		free(dn->dests[i]);
		free(dn->destoffsets[i]);
		free(dn->destlens[i]);
		free(dn->sendblocks[i]);
		free(dn->recvblocks[i]);
	}
	free(dn->sendblocks);
	free(dn->recvblocks);
	free(dn->destlenstot);
	free(dn);
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

	if (commrank == 0) {
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
	}

	/* Test rank partitions */
	int testcommsize = 3;
	idx_t testnumpoints = 4;
	for (idx_t r=0; r<testcommsize; r++)
		printf("Num on rank %lu: %lu\n", r,
				dnf_maxnode(r, testcommsize, testnumpoints));
	idx_t *numperrank = dnf_getlens(testcommsize, testnumpoints);
	for (idx_t r=0; r<testcommsize; r++)
		printf("Num on rank %lu: %lu\n", r, numperrank[r]);
	for (idx_t r=0; r<testcommsize; r++)
		printf("Offset on rank %lu: %lu\n", r,
				dnf_nodeoffset(r, testcommsize, testnumpoints));
	idx_t *startidcs = dnf_getstartidcs(testcommsize, testnumpoints);
	for (idx_t r=0; r<testcommsize; r++)
		printf("Start idx on %lu: %lu\n", r, startidcs[r]);

	/* test delnet from graph */
	unsigned long graph[16] = { [0] = 0, [1] = 2, [2] = 5, [3] = 0,
								[4] = 0, [5] = 0, [6] = 3, [7] = 3,
								[8] = 0, [9] = 0, [10]= 0, [11]= 4,
								[12]= 5, [13]= 0, [14]= 0, [15]= 0 };

	dnf_delaynet *dn = dnf_delaynetfromgraph(graph, 4, commrank, commsize);

	/* take the delnet for a spin */

	/* clean up */
	dnf_freedelaynet(dn, commsize);
	MPI_Finalize();

	return 0;
}
