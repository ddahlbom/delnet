#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "delnetfixed.h"

#define DEBUG 0

/*
   TO DO
   ----
   - [X] Make it so each process communicates *only* with processes
   		 to which it is connected instead of all no matter what.
   - [X] Remove self-communication, profile results (~15-20% improvement)
   - [ ] Debug work partitioning when n not divisible by commsize
   - [X] Optimize buffers (only cycle through necessary number) (~some improvement)
*/


/* --------------- Buffer Functions  ---------------*/
inline dnf_error dnf_bufinit(dnf_delaybuf *buf, unsigned short len)
{
	buf->delaylen = len;
	buf->numstored = 0;
	for (idx_t i=0; i<DNF_BUF_SIZE; i++) 
		buf->counts[i] = 0;
	return DNF_SUCCESS;
}


inline dnf_error dnf_recordevent(dnf_delaybuf *buf)
{
	if (buf->numstored < DNF_BUF_SIZE) {
		buf->counts[buf->numstored] = buf->delaylen;
		buf->numstored += 1;
		return DNF_SUCCESS;
	}
	return DNF_BUFFER_OVERFLOW;
}


inline dnf_error dnf_bufadvance(dnf_delaybuf *buf, data_t *out)
{
	*out = 0.0;
	unsigned short i;
	if (buf->counts[0] == 1) {
		buf->counts[0] = 0;
		*out = 1.0;
		for (i=1; i<buf->numstored; i++)
			buf->counts[i-1] = buf->counts[i];
		buf->numstored -= 1;
	}
	for (i=0; i<buf->numstored; i++) {
		buf->counts[i] -= 1;
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


/* ----- List utils for delaynet initialization only ----- */

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
MPI_Request *sr_counts;
MPI_Request *sr;
MPI_Request *rr_counts;
MPI_Request *rr;
idx_t *outcounts; 	// adjust this (and corresponding above) later so only dn->numoutranks long
idx_t *incounts; 	// adjust this (and corresponding above) later so only dn->numinranks long


void dnf_pushevents(dnf_delaynet *dn, idx_t *eventnodes, idx_t numevents,
					int commrank, int commsize)
{
	for (idx_t i=0; i<commsize; i++)
		outcounts[i] = 0;

	/* populate send blocks */
	/* Note: little difference between having r or n as outer loop var */
	if (DEBUG) printf("Rank %d: Loading send blocks\n", commrank);
	idx_t r;
	for (idx_t i=0; i<dn->numoutranks; i++) {
		r = dn->outranks[i];
		for (idx_t n=0; n<numevents; n++) {
			for (idx_t i=0; i<dn->destlens[r][eventnodes[n]]; i++) {
				dn->sendblocks[r][outcounts[r]] =
					dn->dests[r][dn->destoffsets[r][eventnodes[n]]+i];
				outcounts[r]++;
			}
		}
	}

	/* non-blocking send */
	if (DEBUG) printf("Rank %d: Loaded send blocks. Sending targets.\n", commrank);
	for (idx_t i=0; i<dn->numoutranks; i++) {
		r = dn->outranks[i];
		if (r != commrank) {
			MPI_Isend(&outcounts[r], 1, mpi_idx_t, r,
						0, MPI_COMM_WORLD, &sr_counts[r]);
			MPI_Isend(dn->sendblocks[r], outcounts[r], mpi_idx_t, r,
						1, MPI_COMM_WORLD, &sr[r]);
		}
	}

	/* Receives */
	if (DEBUG) {
		printf("Rank %d: Sent targets. ", commrank);
		printf("Receiving counts and targets.\n");
	}

	for (idx_t i=0; i<dn->numinranks; i++) { 
		r = dn->inranks[i];
		if (r != commrank) 
			MPI_Irecv(&incounts[r], 1, mpi_idx_t, r,
						0, MPI_COMM_WORLD, &rr_counts[r]);
	}

	for (idx_t i=0; i<dn->numinranks; i++) {
		r = dn->inranks[i];
		if (r != commrank) 
			MPI_Wait(&rr_counts[r], MPI_STATUS_IGNORE);
	}

	for (idx_t i=0; i<dn->numinranks; i++) {
		r = dn->inranks[i];
		if (r != commrank) 
			MPI_Irecv(dn->recvblocks[r], incounts[r], mpi_idx_t, r,
						1, MPI_COMM_WORLD, &rr[r]);
	}

	for (idx_t i=0; i<dn->numinranks; i++) {
		r = dn->inranks[i];
		if (r != commrank) 
			MPI_Wait(&rr[r], MPI_STATUS_IGNORE);
	}


	/* record events */
	if (DEBUG) {
		printf("Rank %d: Received counts and targets.", commrank);
		printf("Recording buffer events.\n");
	}

	dnf_error e;
	for (idx_t i=0; i<dn->numinranks; i++) {
		r = dn->inranks[i];
		if (r != commrank) {
			for (idx_t n=0; n<incounts[r]; n++) {
				e = dnf_recordevent(&dn->buffers[dn->recvblocks[r][n]]);
				if (e == DNF_BUFFER_OVERFLOW)
					printf("Buffer full!\n");
			}
		}
		else {
			for (idx_t n=0; n<outcounts[commrank]; n++) {
				e = dnf_recordevent(&dn->buffers[dn->sendblocks[commrank][n]]);
				if (e == DNF_BUFFER_OVERFLOW)
					printf("Buffer full!\n");
			}
		}
	}

	/* wait for sends to finish */
	if (DEBUG) {
		printf("Rank %d: Recorded buffer events.", commrank);
		printf("Ensuring sends complete.\n");
	}
	for (idx_t i=0; i<dn->numoutranks; i++) {
		r = dn->outranks[i];
		if (r != commrank) {
			MPI_Wait(&sr[r], MPI_STATUS_IGNORE);
			MPI_Wait(&sr_counts[r], MPI_STATUS_IGNORE);
		}
	}

	if (DEBUG) printf("Rank %d: Sends complete. Finishing.\n", commrank);
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



int cmp(const void * a, const void * b)
{
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

	/* Share information */ 

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
	dn->recvlenstot = malloc(sizeof(idx_t)*commsize);

	for (idx_t r=0; r<commsize; r++) {
		dn->sendblocks[r] = malloc(sizeof(idx_t)*dn->destlenstot[r]);
		dn->recvlenstot[r] = destlenstot[r];
		dn->recvblocks[r] = malloc(sizeof(idx_t)*destlenstot[r]);
	}


	/* Determine which ranks communicate with each other so can run
	 * communication efficiently */
	dn->numoutranks = 0;
	dn->outranks = malloc(sizeof(idx_t)*commsize);
	for (idx_t r=0; r<commsize; r++) {
		if (dn->destlenstot[r] > 0) {
			dn->outranks[dn->numoutranks] = r;
			dn->numoutranks += 1;
		}
	}
	dn->outranks = realloc(dn->outranks, sizeof(idx_t)*dn->numoutranks);

	dn->numinranks = 0;
	dn->inranks = malloc(sizeof(idx_t)*commsize);
	for (idx_t r=0; r<commsize; r++) {
		if (dn->recvlenstot[r] > 0) {
			dn->inranks[dn->numinranks] = r;
			dn->numinranks += 1;
		}
	}
	dn->inranks = realloc(dn->inranks, sizeof(idx_t)*dn->numinranks);


}


/*
 * Un-optimized delnet initialization function
 */
dnf_delaynet *dnf_delaynetfromgraph(unsigned long *graph, unsigned long n,
									int commrank, int commsize)
{
	dnf_delaynet *dn = malloc(sizeof(dnf_delaynet));
	dn->commrank = commrank;
	dn->commsize = commsize;

	if (n < commsize) {
		printf("Must have fewer processes than nodes!\n Exiting.\n");
		exit(-1);
	}

	/* Establish node parititioning across ranks */
	idx_t *startidcs;
	if (commrank==0) 
		startidcs = dnf_getstartidcs(commsize, n);
	else 
		startidcs = malloc(sizeof(idx_t)*commsize);
	MPI_Bcast(startidcs, commsize, mpi_idx_t, 0, MPI_COMM_WORLD);

	/* so can reconstruct global node number with local data */
	dn->nodeoffsetglobal = startidcs[commrank];


	/* Build process-local delnet infrastructure */
	idx_t n1, n2;
	idx_t *nodesperrank = dnf_getlens(commsize, n);
	n1 = startidcs[commrank];
	n2 = commrank < commsize - 1 ? startidcs[commrank+1] : n;
	dn->numnodes = n2-n1;


	/* Count number of inputs for each node (change column major indexing) */
	idx_t *numinputs = 0;
	numinputs = calloc(n2-n1, sizeof(idx_t));
	for (idx_t c=n1; c<n2; c++) {
		for (idx_t r=0; r<n; r++) 
			if (graph[r*n+c] != 0)
				numinputs[c-n1] += 1;
	}
	

	/* ----- Initialize delaylines and record their origin ----- */
	idx_t *bufferinputnodes = 0;
	idx_t *bufferdestnodes = 0;
	idx_t *bufferoffsets = 0;
	idx_t numinputstotal = 0;

	bufferoffsets = malloc(sizeof(idx_t)*(n2-n1));
	for (idx_t i=0; i<n2-n1; i++) {
		bufferoffsets[i] = numinputstotal;
		numinputstotal += numinputs[i];
	}

	dn->nodeinputbuf = calloc(numinputstotal, sizeof(data_t));
	dn->buffers = malloc(sizeof(dnf_delaybuf)*numinputstotal);
	dn->numbuffers = numinputs;
	dn->numbufferstotal = numinputstotal;
	dn->nodebufferoffsets = bufferoffsets;

	bufferinputnodes = malloc(sizeof(idx_t)*numinputstotal);
	bufferdestnodes = malloc(sizeof(idx_t)*numinputstotal);
	idx_t counter = 0;
	for (idx_t c=n1; c<n2; c++) {
		for (idx_t r=0; r<n; r++) {
			if(graph[r*n+c] != 0) {
				dnf_bufinit(&dn->buffers[counter], graph[r*n+c]);
				bufferinputnodes[counter] = r; 	// ***
				bufferdestnodes[counter] = c;
				counter++;
			}
		}
	}
	dn->buffersourcenodes = bufferinputnodes;
	dn->bufferdestnodes = bufferdestnodes;

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

	idx_t i1, i2, numnodes_l, i_g, runningoffset;
	for (idx_t sd=0; sd<commsize; sd++) {
		i1 = startidcs[sd];
		i2 = sd < commsize - 1 ? startidcs[sd+1] : n;
		numnodes_l = i2-i1;
		if (numnodes_l != nodesperrank[sd]) {printf("Nooo...\n"); exit(-1);}
		destslists[sd] = malloc(sizeof(idx_t*)*numnodes_l);
		destlens[sd] = malloc(sizeof(idx_t)*numnodes_l);
		destoffsets[sd] = malloc(sizeof(idx_t)*numnodes_l);
		runningoffset = 0;
		for (idx_t i=0; i<i2-i1; i++) {
			i_g = i+i1;
			destoffsets[sd][i] = runningoffset;
			destlens[sd][i] = nodedestinations[i_g]->count;
			destslists[sd][i] = malloc(sizeof(idx_t)*destlens[sd][i]);
			for (idx_t j=0; j<destlens[sd][i]; j++)
				/* note - i1 -- so in process local indexing */
				destslists[sd][i][j] = dnf_idxlist_pop(nodedestinations[i_g]); 
			//ASSERT 
			if (nodedestinations[i_g]->count != 0) {
				printf("Assertion Err: didn't finish popping destinations!\n");
				exit(-1);
			}
			dnf_idxlist_free(nodedestinations[i_g]);
			runningoffset += destlens[sd][i];
		}
		destlenstot[sd] = runningoffset;
	}

	free(nodedestinations);

	/* Consolidate list of destinations into master array */
	idx_t **dests = malloc(sizeof(idx_t*)*commsize);
	for (idx_t sd = 0; sd<commsize; sd++) {
		dests[sd] = malloc(sizeof(idx_t)*destlenstot[sd]);
		i1 = startidcs[sd];
		i2 = sd < commsize-1 ? startidcs[sd+1] : n;
		numnodes_l = i2-i1;
		idx_t count = 0;
		for (idx_t i=i1; i<i2; i++) {
			for (idx_t j=0; j<destlens[sd][i-i1]; j++) {
				dests[sd][count] = destslists[sd][i-i1][j];
				count++;
			}
			if (destlens[sd][i-i1] > 0)
				free(destslists[sd][i-i1]);
		}
		//ASSERTION -- logic test only
		if (count != destlenstot[sd]) {
			printf("Assertion Err: Bad transfer of desintation array\n");
			printf("count: %lu, destlenstot: %lu\n", count, destlenstot[sd]);
			exit(-1);
		}
		free(destslists[sd]);
	}
	free(destslists);

	/* Set up data for receiving target info */
	dfn_synctargetinfo(destoffsets, dests, destlens, destlenstot,
					   dn, nodesperrank, commsize);


	/* Coordinate global buffer info */
	idx_t *numbuffersglobal = malloc(sizeof(idx_t)*commsize);
	MPI_Allgather(&dn->numbufferstotal, 1, MPI_UNSIGNED_LONG,
				  numbuffersglobal, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
	dn->bufferoffsetglobal = 0;
	for (idx_t r=1; r<commrank; r++)
		dn->bufferoffsetglobal += numbuffersglobal[r-1];	
	free(numbuffersglobal);



	/* initialize memory for static pointers */
	sr_counts = malloc(sizeof(MPI_Request)*commsize);
	sr = malloc(sizeof(MPI_Request)*commsize);
	rr_counts = malloc(sizeof(MPI_Request)*commsize);
	rr = malloc(sizeof(MPI_Request)*commsize);
	outcounts = calloc(commsize, sizeof(idx_t)); // remove this allocation
	incounts = calloc(commsize, sizeof(idx_t)); // remove this allocation


	/* Clean up remaining unused allocations */
	free(startidcs);
	free(nodesperrank);
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


void dnf_freedelaynet(dnf_delaynet *dn)
{
	free(dn->nodeinputbuf);
	free(dn->nodebufferoffsets);
	free(dn->numbuffers);
	free(dn->buffersourcenodes);
	free(dn->bufferdestnodes);
	free(dn->buffers);
	free(dn->outranks);
	free(dn->inranks);
	free(dn->destlenstot);
	free(dn->recvlenstot);

	for (idx_t i=0; i<dn->commsize; i++) {
		free(dn->dests[i]);
		free(dn->destoffsets[i]);
		free(dn->destlens[i]);
		free(dn->sendblocks[i]);
		free(dn->recvblocks[i]);
	}
	free(dn->dests);
	free(dn->destoffsets);
	free(dn->destlens);
	free(dn->sendblocks);
	free(dn->recvblocks);

	free(dn);

	/* free memory pointed to by static pointers */
	free(outcounts);
	free(incounts);
	free(sr_counts);
	free(sr);
	free(rr_counts);
	free(rr);
}

/* -------------------- IO -------------------- */
void dnf_save(dnf_delaynet *dn, FILE *stream)
{
	fwrite(&dn->commsize, sizeof(int), 1, stream);
	fwrite(&dn->commrank, sizeof(int), 1, stream);
	fwrite(&dn->numnodes, sizeof(idx_t), 1, stream);
	fwrite(&dn->nodeoffsetglobal, sizeof(idx_t), 1, stream);
	fwrite(&dn->bufferoffsetglobal, sizeof(idx_t), 1, stream);

	fwrite(&dn->numbufferstotal, sizeof(idx_t), 1, stream);
	fwrite(dn->nodeinputbuf, sizeof(data_t), dn->numbufferstotal, stream);
	fwrite(dn->nodebufferoffsets, sizeof(idx_t), dn->numnodes, stream);
	fwrite(dn->numbuffers, sizeof(idx_t), dn->numnodes, stream);
	fwrite(dn->buffersourcenodes, sizeof(idx_t), dn->numbufferstotal, stream);
	fwrite(dn->bufferdestnodes, sizeof(idx_t), dn->numbufferstotal, stream);
	fwrite(dn->buffers, sizeof(dnf_delaybuf), dn->numbufferstotal, stream);
	fwrite(&dn->numoutranks, sizeof(idx_t), 1, stream);
	fwrite(dn->outranks, sizeof(idx_t), dn->numoutranks, stream);
	fwrite(&dn->numinranks, sizeof(idx_t), 1, stream);
	fwrite(dn->inranks, sizeof(idx_t), dn->numinranks, stream);
	fwrite(dn->destlenstot, sizeof(idx_t), dn->commsize, stream);
	fwrite(dn->recvlenstot, sizeof(idx_t), dn->commsize, stream);

	for (idx_t r=0; r<dn->commsize; r++) 
		fwrite(dn->destlens[r], sizeof(idx_t), dn->numnodes, stream);
	for (idx_t r=0; r<dn->commsize; r++)
		fwrite(dn->destoffsets[r], sizeof(idx_t), dn->numnodes, stream);
	for (idx_t r=0; r<dn->commsize; r++)
		fwrite(dn->dests[r], sizeof(idx_t), dn->destlenstot[r], stream);

}

dnf_delaynet *dnf_load(FILE *stream)
{
	size_t loadsize;
	dnf_delaynet *dn;
	dn = malloc(sizeof(dnf_delaynet));

	loadsize = fread(&dn->commsize, sizeof(int), 1, stream);
	if (loadsize != 1) { printf("Failed to load dn->commsize\n"); exit(-1); }

	loadsize = fread(&dn->commrank, sizeof(int), 1, stream);
	if (loadsize != 1) { printf("Failed to load dn->commrank\n"); exit(-1); }

	loadsize = fread(&dn->numnodes, sizeof(idx_t), 1, stream);
	if (loadsize != 1) { printf("Failed to load dn->numnodes\n"); exit(-1); }

	loadsize = fread(&dn->nodeoffsetglobal, sizeof(idx_t), 1, stream);
	if (loadsize != 1) {
		printf("Failed to load dn->nodeoffsetglobal\n");
		exit(-1);
	}

	loadsize = fread(&dn->bufferoffsetglobal, sizeof(idx_t), 1, stream);
	if (loadsize != 1) {
		printf("Failed to load dn->bufferoffsetglobal\n");
		exit(-1);
	}

	loadsize = fread(&dn->numbufferstotal, sizeof(idx_t), 1, stream);
	if (loadsize != 1) {
		printf("Failed to load dn->numbufferstotal\n");
		exit(-1);
	}

	dn->nodeinputbuf = malloc(sizeof(data_t)*dn->numbufferstotal);
	loadsize = fread(dn->nodeinputbuf, sizeof(data_t),
					 dn->numbufferstotal, stream);
	if (loadsize != dn->numbufferstotal) {
		printf("Failed to load dn->nodeinputbuf\n");
		exit(-1);
	}

	dn->nodebufferoffsets = malloc(sizeof(idx_t)*dn->numnodes);
	loadsize = fread(dn->nodebufferoffsets, sizeof(idx_t),
					 dn->numnodes, stream);
	if (loadsize != dn->numnodes) {
		printf("Failed to load dn->nodebufferoffsets\n");
		exit(-1);
	}

	dn->numbuffers = malloc(sizeof(idx_t)*dn->numnodes);
	loadsize = fread(dn->numbuffers, sizeof(idx_t), dn->numnodes, stream);
	if (loadsize != dn->numnodes) {
		printf("Failed to load dn->numbuffers\n");
		exit(-1);
	}

	dn->buffersourcenodes = malloc(sizeof(idx_t)*dn->numbufferstotal);
	loadsize = fread(dn->buffersourcenodes, sizeof(idx_t),
					 dn->numbufferstotal, stream);
	if (loadsize != dn->numbufferstotal) {
		printf("Failed to load dn->buffersourcenodes\n");
		exit(-1);
	}

	dn->bufferdestnodes = malloc(sizeof(idx_t)*dn->numbufferstotal);
	loadsize = fread(dn->bufferdestnodes, sizeof(idx_t),
					 dn->numbufferstotal, stream);
	if (loadsize != dn->numbufferstotal) {
		printf("Failed to load dn->bufferdestnodes\n");
		exit(-1);
	}

	dn->buffers = malloc(sizeof(dnf_delaybuf)*dn->numbufferstotal);	
	loadsize = fread(dn->buffers, sizeof(dnf_delaybuf),
					 dn->numbufferstotal, stream);
	if (loadsize != dn->numbufferstotal) {
		printf("Failed to load dn->numbufferstotal\n");
		exit(-1);
	}

	loadsize = fread(&dn->numoutranks, sizeof(idx_t), 1, stream);
	if (loadsize != 1) {
		printf("Failed to load dn->numoutranks\n");
		exit(-1);
	}

	dn->outranks = malloc(sizeof(idx_t)*dn->numoutranks);
	loadsize = fread(dn->outranks, sizeof(idx_t), dn->numoutranks, stream);
	if (loadsize != dn->numoutranks) {
		printf("Failed to load dn->outranks\n");
		exit(-1);
	}

	loadsize = fread(&dn->numinranks, sizeof(idx_t), 1, stream);
	if (loadsize != 1) {
		printf("Failed to load dn->numinranks\n");
		exit(-1);
	}

	dn->inranks = malloc(sizeof(idx_t)*dn->numinranks);
	loadsize = fread(dn->inranks, sizeof(idx_t), dn->numinranks, stream);
	if (loadsize != dn->numinranks) {
		printf("Failed to load dn->inranks\n");
		exit(-1);
	}

	dn->destlenstot = malloc(sizeof(idx_t)*dn->commsize);
	loadsize = fread(dn->destlenstot, sizeof(idx_t), dn->commsize, stream);
	if (loadsize != dn->commsize) {
		printf("Failed to load dn->destlenstot\n");
		exit(-1); 
	}

	dn->recvlenstot = malloc(sizeof(idx_t)*dn->commsize);
	loadsize = fread(dn->recvlenstot, sizeof(idx_t), dn->commsize, stream);
	if (loadsize != dn->commsize) {
		printf("Failed to load dn->recvlenstot\n");
		exit(-1); 
	}

	dn->destlens = malloc(sizeof(idx_t *)*dn->commsize);
	for (idx_t r=0; r<dn->commsize; r++) {
		dn->destlens[r] = malloc(sizeof(idx_t)*dn->numnodes);
		loadsize = fread(dn->destlens[r], sizeof(idx_t), dn->numnodes, stream);
		if (loadsize != dn->numnodes) {
			printf("Failed to load dn->destlens[%lu]\n", r);
			exit(-1);
		}
	}

	dn->destoffsets = malloc(sizeof(idx_t *)*dn->commsize);
	for (idx_t r=0; r<dn->commsize; r++) {
		dn->destoffsets[r] = malloc(sizeof(idx_t)*dn->numnodes);
		loadsize = fread(dn->destoffsets[r], sizeof(idx_t),
						 dn->numnodes, stream);
		if (loadsize != dn->numnodes) {
			printf("Failed to load dn->destloffsets[%lu]\n", r);
			exit(-1);
		}
	}

	dn->dests = malloc(sizeof(idx_t *)*dn->commsize);
	for (idx_t r=0; r<dn->commsize; r++) {
		dn->dests[r] = malloc(sizeof(idx_t)*dn->destlenstot[r]);
		loadsize = fread(dn->dests[r], sizeof(idx_t),
						 dn->destlenstot[r], stream);
		if (loadsize != dn->destlenstot[r]) {
			printf("Failed to load dn->destlenstot[%lu]\n", r);
			exit(-1);
		}
	}
	
	/* initialize static memory */
	sr_counts = malloc(sizeof(MPI_Request)*dn->commsize);
	sr = malloc(sizeof(MPI_Request)*dn->commsize);
	rr_counts = malloc(sizeof(MPI_Request)*dn->commsize);
	rr = malloc(sizeof(MPI_Request)*dn->commsize);
	outcounts = calloc(dn->commsize, sizeof(idx_t)); // remove this allocation
	incounts = calloc(dn->commsize, sizeof(idx_t)); // remove this allocation

	/* allocate recv and send blocks */
	dn->sendblocks = malloc(sizeof(idx_t *)*dn->commsize);
	dn->recvblocks = malloc(sizeof(idx_t *)*dn->commsize);
	for (idx_t r=0; r<dn->commsize; r++) {
		dn->sendblocks[r] = malloc(sizeof(idx_t)*dn->destlenstot[r]);
		dn->recvblocks[r] = malloc(sizeof(idx_t)*dn->recvlenstot[r]);
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

//int main(int argc, char *argv[]) 
//{
//
//	/* Init MPI */
//	int commsize, commrank;
//	MPI_Init(&argc, &argv);	
//	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
//	MPI_Comm_rank(MPI_COMM_WORLD, &commrank);
//
//	if (commrank == 0) {
//		dnf_delaybuf buf;
//		dnf_bufinit(&buf, 6);
//		data_t output = 0.0;
//
//		idx_t eventtimes[] = {3, 11, 18, 54, 76, 92};
//		idx_t n = 6;
//
//		for (idx_t i=0; i<100; i++) {
//			printf("Step %lu: %lf\n", i, output);
//			if (in(i, eventtimes, n))
//				dnf_recordevent(&buf, (double) i);
//			dnf_bufadvance(&buf, &output);
//		}
//// test partitioning 
//		idx_t numpoints = 1000;
//		idx_t numranks = 7;
//		idx_t *startidcs = dnf_getstartidcs(numranks, numpoints);
//		for (int i=0; i<numranks; i++)
//			printf("Start index on rank %d: %lu\n", i, startidcs[i]);
//		free(startidcs);
//	}
//
//	/* Test rank partitions */
//	int testcommsize = 3;
//	idx_t testnumpoints = 4;
//	for (idx_t r=0; r<testcommsize; r++)
//		printf("Num on rank %lu: %lu\n", r,
//				dnf_maxnode(r, testcommsize, testnumpoints));
//	idx_t *numperrank = dnf_getlens(testcommsize, testnumpoints);
//	for (idx_t r=0; r<testcommsize; r++)
//		printf("Num on rank %lu: %lu\n", r, numperrank[r]);
//	for (idx_t r=0; r<testcommsize; r++)
//		printf("Offset on rank %lu: %lu\n", r,
//				dnf_nodeoffset(r, testcommsize, testnumpoints));
//	idx_t *startidcs = dnf_getstartidcs(testcommsize, testnumpoints);
//	for (idx_t r=0; r<testcommsize; r++)
//		printf("Start idx on %lu: %lu\n", r, startidcs[r]);
//
//	/* test delnet from graph */
//	unsigned long graph[16] = { [0] = 0, [1] = 2, [2] = 5, [3] = 0,
//								[4] = 0, [5] = 0, [6] = 3, [7] = 3,
//								[8] = 0, [9] = 0, [10]= 0, [11]= 4,
//								[12]= 5, [13]= 0, [14]= 0, [15]= 0 };
//
//	dnf_delaynet *dn = dnf_delaynetfromgraph(graph, 4, commrank, commsize);
//
//	/* take the delnet for a spin */
//	char display[1024];
//	idx_t events[100] = {0};
//	idx_t numevents = 0;
//	data_t *input;
//
//	dnf_recordevent(&dn->buffers[0], 0.0); // give it an initial kick
//	for (idx_t i=0; i<100; i++) {
//		MPI_Barrier(MPI_COMM_WORLD);
//		display[0] = '\0';
//		if (commrank == 0) printf("\nIteration %lu\n", i);
//		dnf_advance(dn);
//		for (idx_t n=0; n<dn->numnodes; n++) {
//			input = dnf_getinputaddress(dn, n);
//			for (idx_t j=0; j<dn->numbuffers[n]; j++) {
//				if (input[j] != 0) {
//					events[numevents] = n;
//					numevents += 1;
//					break;
//				}
//			}
//		}
//
//		sprintf(display, "(Rank %d) Num events: %lu\n", commrank, numevents);
//		strcat(display, "\t");
//		for (idx_t i=0; i<numevents; i++)
//			sprintf(display + strlen(display), "%lu ", events[i]);
//		strcat(display, "\n");
//		printf("%s", display);
//
//		dnf_pushevents(dn, events, numevents, commrank, commsize, (double) i);
//		numevents=0;
//	}
//
//
//	/* clean up */
//	dnf_freedelaynet(dn);
//	MPI_Finalize();
//
//	return 0;
//}
//
//
