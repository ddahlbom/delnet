#include <stdlib.h>
#include <stdio.h> 	//for profiling/debugging
#include <time.h> 	//for profiling/debugging
#include <math.h>
// #include <mpi.h> 

//#include "/usr/include/mpich/mpi.h"
#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#include "delnetmpi.h"

#define DEBUG 0

#define OFFSET 0
#define LENGTH 1
#define DATA 2


/* for sorting */
int cmpfunc (const void * a, const void * b) {
   return ( *(long int*)a - *(long int*)b );
}

/* --------------------	MPI Utils -------------------- */
size_t dn_mpi_maxnode(int rank, int commsize, size_t numpoints)
{
	size_t basesize = floor(numpoints/commsize);
	return rank < (numpoints % commsize) ? basesize + 1 : basesize;
}

size_t dn_mpi_nodeoffset(int rank, int commsize, size_t numpoints)
{
	size_t offset = 0; int i=0;
	while (i < rank) {
		offset += dn_mpi_maxnode(rank, commsize, numpoints);
		i++;
	}
	return offset;
}


/*
 * -------------------- Util Functions --------------------
 */

dn_mpi_list_uint *dn_mpi_list_uint_init() {
	dn_mpi_list_uint *newlist;
	newlist = malloc(sizeof(dn_mpi_list_uint));
	newlist->count = 0;
	newlist->head = NULL; 

	return newlist;
}

void dn_mpi_list_uint_push(dn_mpi_list_uint *l, unsigned int val) {
	dn_mpi_listnode_uint *newnode;
	newnode = malloc(sizeof(dn_mpi_listnode_uint));
	newnode->val = val;
	newnode->next = l->head;
	l->head = newnode;
	l->count += 1;
}

unsigned int dn_mpi_list_uint_pop(dn_mpi_list_uint *l) {
	unsigned int val;
	dn_mpi_listnode_uint *temp;

	if (l->head != NULL) {
		val = l->head->val;
		temp = l->head;
		l->head = l->head->next;
		free(temp);
		l->count -= 1;
	}
	else {
		// trying to pop an empty list
		//val = 0;
		exit(-1);
	}
	return val;
}

void dn_mpi_list_uint_free(dn_mpi_list_uint *l) {
	while (l->head != NULL) {
		dn_mpi_list_uint_pop(l);
	}
	free(l);
}


dn_mpi_vec_float dn_mpi_orderbuf(IDX_T which, dn_mpi_delaynet *dn) {
	IDX_T k, n, idx;
	dn_mpi_vec_float output;
	
	n = dn->del_lens[which];
	output.n = n;
	output.data = malloc(sizeof(FLOAT_T) * n);

	for (k=0; k<n; k++) {
		idx = dn->del_startidces[which] +
			((dn->del_offsets[which]+k) % dn->del_lens[which]);
		output.data[n-k-1] = dn->delaybuf[idx];
	}
	return output;
}

char *dn_mpi_vectostr(dn_mpi_vec_float input) {
	int k;
	char *output;
	output = malloc(sizeof(char)*(input.n+1));
	output[input.n] = '\0';
	for(k=0; k < input.n; k++) {
		output[k] = input.data[k] == 0.0 ? '-' : '1';
	}
	return output;
}



/*
 * -------------------- delnet Functions --------------------
 */

/* push output of nodes into delay network (input of dn) */
void dn_mpi_pushoutput(FLOAT_T val, IDX_T idx, dn_mpi_delaynet *dn) 
{
	IDX_T i1, i2, k;

	i1 = dn->nodes[idx].idx_inbuf;
	i2 = i1 + dn->nodes[idx].num_out;

	for (k = i1; k < i2; k++)
		dn->inputs[k] = val;
}


/* No getinputs()... would need to return vector */
/*
dn_mpi_vec_float dn_mpi_getinputvec(dn_mpi_delaynet *dn) {
	dn_mpi_vec_float inputs;	
	inputs.data = malloc(sizeof(FLOAT_T)*dn->num_delays);
	inputs.n = dn->num_delays;
	for (int i=0; i<dn->num_delays; i++) {
		inputs.data[i] = dn->inputs[i];
	}
	return inputs;
}
*/

/* get inputs to neurons (outputs of delaynet)... */
FLOAT_T *dn_mpi_getinputaddress(IDX_T idx, dn_mpi_delaynet *dn) {
	return &dn->outputs_l[dn->nodes[idx].idx_outbuf];
}


void dn_mpi_advance(dn_mpi_delaynet *dn)
{
	IDX_T i, j, k, k_global;
	size_t offset;

	if (DEBUG)  printf("Entered advancement function\n");

	/* load network inputs into buffers */
	//for(k=0; k < dn->numlinesin_l; k++) {
	for(k=0; k < dn->numlinesout_l; k++) {
		dn->delaybuf[dn->del_startidces[k]+dn->del_offsets[k]] = dn->inputs[k];
	}

	/* advance the buffers */
	for(k=0; k < dn->numlinesout_l; k++) {
		dn->del_offsets[k] = (dn->del_offsets[k] + 1) % dn->del_lens[k];
	}

	/* pull network outputs from buffers */
	//for (k=0; k < dn->numlinesout_l; k++) {
	for (k=0; k < dn->numlinesout_l; k++) {
		k_global = k + dn->lineoffset_in; 	// because still in input buffer order
		//dn->outputs_unsorted[k_global] =
		dn->outputs_unsorted[k] =
				dn->delaybuf[dn->del_startidces[k]+
								dn->del_offsets[k]];
	}

	/* Collect blocks to send to each rank */	
	for (i=0; i<dn->commsize; i++) {
		for (j=0; j<dn->out_unsorted_blocklens[i]; j++) {
			dn->out_unsorted_blocks[i][j] = dn->out_unsorted_l[ dn->block_idcs[i][j] ];
		}
	}

	/* transfer local block to out_unsorted_l */
	offset = 0;
	for (j=0; j<dn->commrank; j++) offset += dn->incoming_blocklens[j];

	for (i=0; i<dn->out_unsorted_blocklens[dn->commrank]; i++) {
		dn->out_unsorted_l[i+offset] = dn->out_unsorted_blocks[dn->commrank][i];
	}


	/*
	 * -------------------- MPI DATA TRANSFER -------------------- 
	 */

	MPI_Request dataRecvReq[dn->commsize];
	MPI_Request dataSendReq[dn->commsize];

	/* Nonblocking receives */
	for (i=0; i<dn->commsize; i++) {
		if (i != dn->commrank) {
			
			/* do these offsets in advance later */
			offset = 0;
			for (j=0; j<i; j++) offset += dn->incoming_blocklens[j];

			MPI_Irecv(&dn->out_unsorted_l[offset],
					  dn->incoming_blocklens[i], //<--- This isn't known!!!
					  MPI_DOUBLE,
					  i,
					  DATA,
					  MPI_COMM_WORLD,
					  &dataRecvReq[i]);
		}
	}

	/* Nonblocking sends */
	for (i=0; i<dn->commsize; i++) {
		if (i != dn->commrank) {
			MPI_Isend(&dn->out_unsorted_blocks[i],
					  dn->out_unsorted_blocklens[i],
					  MPI_DOUBLE,
					  i,
					  DATA,
					  MPI_COMM_WORLD,
					  &dataSendReq[i]);
		}
	}

	// Wait to receive data information
	for (i=0; i<dn->commsize; i++) {
		if (i!=dn->commrank) {
			if (DEBUG) printf("Waiting on data from %d on %d\n", i, dn->commrank);
			MPI_Wait(&dataRecvReq[i], MPI_STATUS_IGNORE);
		}
	}

	// Wait for sent data to be received 
	for (i=0; i<dn->commsize; i++) {
		if (i!=dn->commrank) {
			if (DEBUG) printf("Waiting for data to be recieved on %d from %d\n",i,dn->commrank);
			MPI_Wait(&dataSendReq[i], MPI_STATUS_IGNORE);
		}
	}

	/*
	 * -------------------- END MPI DATA TRANSFER -------------------- 
	 */
	MPI_Barrier(MPI_COMM_WORLD);


	/* sort output for access */
	for (k=0; k< dn->numlinesin_l; k++) {
		dn->outputs_l[k] = dn->out_unsorted_l[ dn->sourceidx_l[k] ];
	}
	if (DEBUG) printf("Exiting advancing step...\n");
}


unsigned int *dn_mpi_blobgraph(unsigned int n, float p, unsigned int maxdel) {
	unsigned int count = 0;
	unsigned int *delmat;
	unsigned int i, j;
	delmat = malloc(sizeof(int)*n*n);

	for (i=0; i<n; i++) 
	for (j=0; j<n; j++) {
		if (unirand() < p && i != j) {
			delmat[i*n+j] = getrandom(maxdel) + 1;
			count += 1;
		}
		else 
			delmat[i*n+j] = 0;
	}

	return delmat;
}


dn_mpi_delaynet *dn_mpi_delnetfromgraph(unsigned int *g, unsigned int n, 
											int commrank, int commsize)
{
	unsigned int i, j, delcount, startidx;
	unsigned int deltot_g, numlines_g, deltot_l, numlinesout_l, numlinesin_l;
	size_t num_nodes_l = dn_mpi_maxnode(commrank, commsize, n);
	size_t nodeoffset = dn_mpi_nodeoffset(commrank, commsize, n);

	dn_mpi_delaynet *dn;
	dn_mpi_list_uint **nodes_in;

	dn = malloc(sizeof(dn_mpi_delaynet));
	nodes_in = malloc(sizeof(dn_mpi_list_uint)*n);
	for (i=0; i<n; i++)
		nodes_in[i] = dn_mpi_list_uint_init();

	/* analyze how much memory to allocate */
	deltot_g = 0;
	numlines_g = 0;
	deltot_l = 0;
	numlinesout_l = 0;
	numlinesin_l = 0;

	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			deltot_g += g[i*n+j];
			numlines_g += g[i*n+j] != 0 ? 1 : 0;
			// Block of rows -- number of lines out
			if (i >= nodeoffset && i < nodeoffset + num_nodes_l) {
				deltot_l += g[i*n+j];
				numlinesout_l += g[i*n+j] != 0 ? 1 : 0;
			}
			// Block of columns -- number of lines in
			if (j >= nodeoffset && j < nodeoffset + num_nodes_l) {
				numlinesin_l += g[i*n+j] != 0 ? 1 : 0;
			}
		}
	}

	dn->num_nodes_g = n;
	dn->num_nodes_l = num_nodes_l;
	dn->nodeoffset = nodeoffset;
	dn->numlinesin_l = numlinesin_l;
	dn->numlinesout_l = numlinesout_l;
	dn->numlines_g = numlines_g;
	dn->buf_len = deltot_l;
	dn->commrank = commrank;
	dn->commsize = commsize;

	dn->delaybuf = calloc(deltot_l, sizeof(FLOAT_T));
	dn->inputs = calloc(numlinesout_l, sizeof(FLOAT_T));
	dn->outputs = calloc(numlinesin_l, sizeof(FLOAT_T)); 	
	dn->outputs_unsorted = calloc(numlines_g, sizeof(FLOAT_T)); // <- ultimately refine this

	// Original:
	// dn->del_offsets = malloc(sizeof(IDX_T)*numlines);
	// dn->del_startidces = malloc(sizeof(IDX_T)*numlines);
	// dn->del_lens = malloc(sizeof(IDX_T)*numlines);
	// dn->del_sources = malloc(sizeof(IDX_T)*numlines);
	// dn->del_targets = malloc(sizeof(IDX_T)*numlines);
	// dn->nodes = malloc(sizeof(dn_mpi_node)*n);
	
	// Local backup of original
	IDX_T *del_offsets = malloc(sizeof(IDX_T)*numlines_g);
	IDX_T *del_startidces = malloc(sizeof(IDX_T)*numlines_g);
	IDX_T *del_lens = malloc(sizeof(IDX_T)*numlines_g);
	IDX_T *del_sources = malloc(sizeof(IDX_T)*numlines_g);
	IDX_T *del_targets = malloc(sizeof(IDX_T)*numlines_g);
	dn_mpi_node *nodes = malloc(sizeof(dn_mpi_node)*n);


	/* init nodes -- should've just calloc-ed*/
	//for (i=nodeoffset; i<nodeoffset+num_nodes_l; i++) {
	for (i=0; i<n; i++) {
		nodes[i].idx_outbuf = 0;
		nodes[i].num_in = 0;
		nodes[i].idx_inbuf = 0;
		nodes[i].num_out = 0;
	}

	/* work through graph, allocate delay lines */
	delcount = 0;
	startidx = 0;
	for (i = 0; i<n; i++)
	for (j = 0; j<n; j++) {
		if (g[i*n + j] != 0) {
			dn_mpi_list_uint_push(nodes_in[j], i);

			del_offsets[delcount] = 0;
			del_startidces[delcount] = startidx;
			del_lens[delcount] = g[i*n+j];
			del_sources[delcount] = i;
			del_targets[delcount] = j;

			nodes[i].num_out += 1;

			startidx += g[i*n+j];
			delcount += 1;
		}
	}
	
	/* work out rest of index arithmetic */
	unsigned int *num_outputs, *in_base_idcs;
	num_outputs = calloc(n, sizeof(unsigned int));
	in_base_idcs = calloc(n, sizeof(unsigned int));
	for (i=0; i<n; i++) {
		num_outputs[i] = nodes[i].num_out;
		for (j=0; j<i; j++)
			in_base_idcs[i] += num_outputs[j]; 	// check logic here
		//in_base_idcs[i] = i == 0 ? 0 : in_base_idcs[i-1] + num_outputs[i];
	}

	unsigned int idx = 0;
	for (i=0; i<n; i++) {
		nodes[i].num_in = nodes_in[i]->count;
		nodes[i].idx_outbuf = idx;
		idx += nodes[i].num_in;
		nodes[i].idx_inbuf = in_base_idcs[i];
	}

	unsigned int *num_inputs, *out_base_idcs, *out_counts, *inverseidces;
	num_inputs = calloc(n, sizeof(unsigned int));
	out_base_idcs = calloc(n, sizeof(unsigned int));
	out_counts = calloc(n, sizeof(unsigned int));
	for (i=0; i<n; i++) {
		num_inputs[i] = nodes[i].num_in;
		for (j=0; j<i; j++)
			out_base_idcs[i] += num_inputs[j]; // check logic here
		//out_base_idcs[i] = i == 0 ? 0 : in_base_idcs[i-1] + num_inputs[i];
	}

	inverseidces = calloc(numlines_g, sizeof(unsigned int));
	for (i=0; i < numlines_g; i++) {
		inverseidces[i] = out_base_idcs[del_targets[i]] + 
						  out_counts[del_targets[i]];
		out_counts[del_targets[i]] += 1;
	}

	dn->destidx_g = inverseidces;
	dn->sourceidx_g = malloc(sizeof(unsigned int)*numlines_g);
	for (i=0; i<numlines_g; i++)
		dn->sourceidx_g[dn->destidx_g[i]] = i;


	/* -- DIVERGES FROM GLOBAL HERE -- */
	/* take just the local info */
	dn->del_offsets = malloc(sizeof(IDX_T)*numlinesout_l);
	dn->del_startidces = malloc(sizeof(IDX_T)*numlinesout_l);
	dn->del_lens = malloc(sizeof(IDX_T)*numlinesout_l);
	dn->del_sources = malloc(sizeof(IDX_T)*numlinesout_l);
	dn->del_targets = malloc(sizeof(IDX_T)*numlinesout_l);
	dn->nodes = malloc(sizeof(dn_mpi_node)*num_nodes_l);

	IDX_T i0_inbuf, i1_inbuf, i0_outbuf, i1_outbuf;
	i0_inbuf = nodes[nodeoffset].idx_inbuf;
	i1_inbuf = nodes[nodeoffset+num_nodes_l-1].idx_inbuf
				+ nodes[nodeoffset+num_nodes_l-1].num_out;
	i0_outbuf = nodes[nodeoffset].idx_outbuf;
	i1_outbuf = nodes[nodeoffset+num_nodes_l-1].idx_outbuf
				+ nodes[nodeoffset+num_nodes_l-1].num_in;

	dn->lineoffset_in = i0_inbuf;
	dn->lineoffset_out = i0_outbuf;

	// ASSERT!
	if (i1_inbuf-i0_inbuf != numlinesout_l) {
		printf("Process %d -- i1_inbuf-i0_inbuf: %d \t numlinesout_l: %d\n",
				commrank, i1_inbuf-i0_inbuf, numlinesout_l);
		printf("You f*!@ed up indexing!\n");
		exit(-1);
	}

	if (i1_outbuf-i0_outbuf != numlinesin_l) {
		printf("Process %d -- i1_outbuf-i0_outbuf: %d \t numlinesin_l: %d\n",
				commrank, i1_outbuf-i0_outbuf, numlinesin_l);
		printf("You f*!@ed up indexing!\n");
		exit(-1);
	}

	for (i=0; i<numlinesout_l; i++) {
		dn->del_offsets[i] = del_offsets[i+i0_inbuf]; // kinda stupid -- they're all zero
		dn->del_startidces[i] = del_startidces[i+i0_inbuf] - del_startidces[i0_inbuf];
		dn->del_lens[i] = del_lens[i+i0_inbuf];
		dn->del_sources[i] = del_sources[i+i0_inbuf];
		dn->del_targets[i] = del_targets[i+i0_inbuf];
	}

	IDX_T in_offset = nodes[nodeoffset].idx_inbuf;
	IDX_T out_offset = nodes[nodeoffset].idx_outbuf;
	for (i=0; i<num_nodes_l; i++) {
		dn->nodes[i].idx_inbuf = nodes[nodeoffset+i].idx_inbuf - in_offset;
		dn->nodes[i].idx_outbuf = nodes[nodeoffset+i].idx_outbuf - out_offset;
		dn->nodes[i].num_out = nodes[nodeoffset+i].num_out;
		dn->nodes[i].num_in = nodes[nodeoffset+i].num_in;
	}

	/* free the unused global info */
	free(del_offsets);
	free(del_startidces);
	free(del_lens);
	free(del_sources);
	free(del_targets);

	/* ----- Diverges from Simple MPI Version -- sort out local blocks ----- */
	/* Now building on previous step... step up local blocks */
	FLOAT_T **out_unsorted_blocks = calloc(commsize, sizeof(FLOAT_T*));
	unsigned int *out_unsorted_blocklens = calloc(commsize, sizeof(unsigned int));
	size_t **block_idcs = calloc(commsize, sizeof(size_t*));
	FLOAT_T *out_unsorted_l = calloc(dn->numlinesin_l, sizeof(FLOAT_T));

	/* initialize counts */
	for (i=0; i<commsize; i++) out_unsorted_blocklens[i] = 0;

	/* count sizes of unsorted blocks */	
	/* go through local section of unsorted outputs, count how many to each -- */

	/* get the boundary indices -- NOTE NODES, LINES!!! */
	/* carefully consider with in or out here -- functioning of sourceidx */
	size_t boundaryidcs[commsize+1];
	size_t nodenum;
	boundaryidcs[0] = 0;
	for (i=1; i<commsize+1; i++) {
		if (i<commsize) {
			nodenum = dn_mpi_nodeoffset(i, commsize, n);
		} else {
			nodenum = n;
		}
		boundaryidcs[i] = nodes[nodenum-1].idx_outbuf + nodes[nodenum-1].num_in; 
	}

	//for (i=1; i<commsize+1; i++) boundaryidcs[i] = boundaryidcs[i] + boundaryidcs[i-1];

	/* determine sizes of blocks to allocate */
	for (i=dn->lineoffset_in; i < dn->lineoffset_in+numlinesout_l; i++) {
		for (j=0; j<commsize; j++) {
			if (boundaryidcs[j] <= dn->sourceidx_g[i]
					&& dn->sourceidx_g[i] < boundaryidcs[j+1]) {
				out_unsorted_blocklens[j] += 1;
			}
		}
	}

	//ASSERTION:
	size_t localblocklen=0;
	for (i=0; i<commsize; i++) localblocklen += out_unsorted_blocklens[i];
	if (localblocklen != numlinesout_l) {
		printf("Assertion error: local blocks don't add up to number of lines\n");
		exit(-1);
	}

	/* ---------- START: MPI TRANSFER INCOMING BLOCK INFO ---------- */
	
	MPI_Request lenRecvReq[commsize]; 	// can make commsize-1; one dummy for indexing
	MPI_Request lenSendReq[commsize]; 	// can make commsize-1

	dn->incoming_blocklens = calloc(commsize, sizeof(size_t));

	/* receive communicate block sizes */
	for (i=0; i<commsize;i++) {
		if (i != commrank) {
			MPI_Irecv(&dn->incoming_blocklens[i],
					  1,
					  MPI_UNSIGNED,
					  i,
					  LENGTH,
					  MPI_COMM_WORLD,
					  &lenRecvReq[i]);
		}
	}
	
	// this is the local block -- just initializing to sensible value
	// shouldn't be used...
	dn->incoming_blocklens[commrank] = out_unsorted_blocklens[commrank]; 	

	/* send block sizes */
	for (i=0; i<commsize;i++) {
		if (i != commrank) {
			MPI_Isend(&out_unsorted_blocklens[i],
					  1,
					  MPI_UNSIGNED,
					  i,
					  LENGTH,
					  MPI_COMM_WORLD,
					  &lenSendReq[i]);
		}
	}

	/* Wait to receive data information */
	for (i=0; i<dn->commsize; i++) {
		if (i!=dn->commrank) {
			if (DEBUG) printf("Waiting on blocklengths from %d on %d\n", i, dn->commrank);
			MPI_Wait(&lenRecvReq[i], MPI_STATUS_IGNORE);
		}
	}

	/* Wait for sent data to be received */
	for (i=0; i<dn->commsize; i++) {
		if (i!=dn->commrank) {
			if (DEBUG) printf("Waiting for blocklengths to be recieved on %d from %d\n",i,dn->commrank);
			MPI_Wait(&lenSendReq[i], MPI_STATUS_IGNORE);
		}
	}

	/* ---------- END: MPI TRANSFER INCOMING BLOCK INFO ---------- */

	//ASSERTION
	unsigned int totalinblocklen = 0;
	for (i=0; i<commsize; i++) totalinblocklen += dn->incoming_blocklens[i];
	if (totalinblocklen != numlinesin_l) {
		printf("Assertion error: Incoming block lengths don't add to number of input lines!\n");
		printf("incoming: %ud, lines in: %ud\n", totalinblocklen, numlinesin_l);
		exit(-1);
	}

	/* Allocate block idces and local blocks to send out */
	for (i=0; i<commsize; i++) {
		block_idcs[i] = 		 calloc(out_unsorted_blocklens[i], sizeof(size_t));
		out_unsorted_blocks[i] = calloc(out_unsorted_blocklens[i], sizeof(FLOAT_T));
	}

	/* Collect idces for each block  */
	size_t *counts = calloc(commsize, sizeof(size_t));
	size_t i_g;
	for (i=0; i < numlinesout_l; i++) {
		i_g = i + dn->lineoffset_in; 	// <-- check..
		for (j=0; j<commsize; j++) {
			if (boundaryidcs[j] <= dn->sourceidx_g[i_g] 
					&& dn->sourceidx_g[i_g] < boundaryidcs[j+1]) {
				//block_idcs[j][counts[j]] = dn->sourceidx_g[i];
				block_idcs[j][counts[j]] = i; //local index
				counts[j]++;
			}
		}
	}
	//ASSERTION
	for (i=0; i<commsize; i++) {
		if (counts[i] != out_unsorted_blocklens[i]) {
			printf("Assertion error: local block indexing off.\n");
			exit(-1);
		}
	}
	free(counts);


	/* Calculate inverse idices for local block */
	long int *sort_idcs = calloc(dn->numlinesin_l, sizeof(long int));
	size_t *sourceidx_l = calloc(dn->numlinesin_l, sizeof(size_t));
	for (i=0; i<dn->numlinesin_l; i++) {
		i_g = i + dn->lineoffset_out;
		sort_idcs[i] = (long int) dn->sourceidx_g[i_g]; 
	}
	// Sort 
	qsort(sort_idcs, dn->numlinesin_l, sizeof(long int), cmpfunc);

	/* Find new local mapping indices
	 * This is grossly inelegant, but (hopefully) functional. It is only done
	 * on initialization, so doesn't effect main calculation. Optimzation can
	 * be done at some future stage in init times are too long */
	size_t sidx;
	for (i=0; i<dn->numlinesin_l; i++) {
		i_g = i + dn->lineoffset_out;
		sidx = dn->sourceidx_g[i_g];
		for (j=0; j<dn->numlinesin_l; j++) {
			if (sort_idcs[j] == sidx) {
				sourceidx_l[i] = j;
				break;
			}
		}
	}
	free(sort_idcs);

	/* put data into delnet structure */
	dn->sourceidx_l = sourceidx_l;
	dn->out_unsorted_blocklens = out_unsorted_blocklens;
	dn->out_unsorted_blocks = out_unsorted_blocks;
	dn->out_unsorted_l = out_unsorted_l;
	dn->outputs_l = calloc(dn->numlinesin_l , sizeof(FLOAT_T));
	dn->block_idcs = block_idcs;

	/* Clean up */
	for (i=0; i<n; i++) dn_mpi_list_uint_free(nodes_in[i]);
	free(nodes_in);
	free(nodes);
	free(num_outputs);
	free(in_base_idcs);
	free(num_inputs);
	free(out_base_idcs);
	free(out_counts);

	return dn;
}

void dn_mpi_freedelnet(dn_mpi_delaynet *dn) {
	free(dn->del_offsets);
	free(dn->del_startidces);
	free(dn->del_lens);
	free(dn->del_sources);
	free(dn->del_targets);
	free(dn->inputs);
	free(dn->outputs);
	free(dn->outputs_unsorted);
	free(dn->destidx_g);
	free(dn->sourceidx_g);
	free(dn->delaybuf);
	free(dn->nodes);

	/* MPI Revisions */
	free(dn->sourceidx_l);
	free(dn->out_unsorted_blocklens);
	for (int i=0; i<dn->commsize; i++) free(dn->out_unsorted_blocks[i]);
	free(dn->out_unsorted_l);
	free(dn->outputs_l); 
	for (int i=0; i<dn->commsize; i++) free(dn->block_idcs[i]);
	
	free(dn);
}


void dn_mpi_savecheckpt(dn_mpi_delaynet *dn, FILE *stream) {
	fwrite(&dn->num_nodes_g, sizeof(IDX_T), 1, stream);
	fwrite(&dn->num_nodes_l, sizeof(IDX_T), 1, stream);
	fwrite(&dn->nodeoffset, sizeof(IDX_T), 1, stream);
	fwrite(&dn->numlinesout_l, sizeof(IDX_T), 1, stream);
	fwrite(&dn->numlinesin_l, sizeof(IDX_T), 1, stream);
	fwrite(&dn->numlines_g, sizeof(IDX_T), 1, stream);
	fwrite(&dn->lineoffset_in, sizeof(IDX_T), 1, stream);
	fwrite(&dn->lineoffset_out, sizeof(IDX_T), 1, stream);
	fwrite(&dn->buf_len, sizeof(IDX_T), 1, stream);
	fwrite(&dn->commrank, sizeof(int), 1, stream);
	fwrite(&dn->commsize, sizeof(int), 1, stream);

	fwrite(dn->delaybuf, sizeof(FLOAT_T), dn->buf_len, stream);
	fwrite(dn->inputs, sizeof(FLOAT_T), dn->numlinesout_l, stream);
	fwrite(dn->outputs, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	fwrite(dn->outputs_unsorted, sizeof(FLOAT_T), dn->numlines_g, stream);
	fwrite(dn->nodes, sizeof(dn_mpi_node), dn->num_nodes_l, stream);
	fwrite(dn->destidx_g, sizeof(IDX_T), dn->numlines_g, stream);
	fwrite(dn->sourceidx_g, sizeof(IDX_T), dn->numlines_g, stream);
	fwrite(dn->del_offsets, sizeof(IDX_T), dn->numlinesout_l, stream);
	fwrite(dn->del_startidces, sizeof(IDX_T), dn->numlinesout_l , stream);
	fwrite(dn->del_lens, sizeof(IDX_T), dn->numlinesout_l, stream);
	fwrite(dn->del_sources, sizeof(IDX_T), dn->numlinesout_l, stream);
	fwrite(dn->del_targets, sizeof(IDX_T), dn->numlinesout_l, stream);

	/* MPI Revisions */
	fwrite(dn->sourceidx_l, sizeof(size_t), dn->numlinesin_l, stream);
	fwrite(dn->out_unsorted_blocklens, sizeof(unsigned int), dn->commsize, stream);
	for (int i=0; i<dn->commsize; i++) {
		fwrite(dn->out_unsorted_blocks[i], sizeof(FLOAT_T),
				dn->out_unsorted_blocklens[i], stream); 
	}
	fwrite(dn->out_unsorted_l, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	fwrite(dn->outputs_l, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	for (int i=0; i<dn->commsize; i++) {
		fwrite(dn->block_idcs, sizeof(size_t),
				dn->out_unsorted_blocklens[i], stream);
	}
}


dn_mpi_delaynet *dn_mpi_loadcheckpt(FILE *stream) {
	size_t loadsize;
	dn_mpi_delaynet *dn;
	dn = malloc(sizeof(dn_mpi_delaynet));

	/* load constants */
	loadsize = fread(&dn->num_nodes_g, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->num_nodes_l, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->nodeoffset, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->numlinesout_l, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->numlinesin_l, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->numlines_g, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->lineoffset_in, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->lineoffset_out, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->buf_len, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->commrank, sizeof(int), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->commsize, sizeof(int), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }

	/* load big chunks */
	dn->delaybuf = malloc(sizeof(FLOAT_T)*dn->buf_len);
	loadsize = fread(dn->delaybuf, sizeof(FLOAT_T), dn->buf_len, stream);
	if (loadsize != dn->buf_len) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->inputs = malloc(sizeof(FLOAT_T)*dn->numlinesout_l);
	loadsize = fread(dn->inputs, sizeof(FLOAT_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }
	
	dn->outputs = malloc(sizeof(FLOAT_T)*dn->numlinesin_l);
	loadsize = fread(dn->outputs, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	if (loadsize != dn->numlinesin_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->outputs_unsorted = malloc(sizeof(FLOAT_T)*dn->numlines_g);
	loadsize = fread(dn->outputs_unsorted, sizeof(FLOAT_T), dn->numlines_g, stream);
	if (loadsize != dn->numlines_g) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->nodes = malloc(sizeof(dn_mpi_node)*dn->num_nodes_l);
	loadsize = fread(dn->nodes, sizeof(dn_mpi_node), dn->num_nodes_l, stream);
	if (loadsize != dn->num_nodes_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->destidx_g = malloc(sizeof(IDX_T)*dn->numlines_g);
	loadsize = fread(dn->destidx_g, sizeof(IDX_T), dn->numlines_g, stream);
	if (loadsize != dn->numlines_g) { printf("Failed to load delay network.\n"); exit(-1); }
	
	dn->sourceidx_g = malloc(sizeof(IDX_T)*dn->numlines_g);
	loadsize = fread(dn->sourceidx_g, sizeof(IDX_T), dn->numlines_g, stream);
	if (loadsize != dn->numlines_g) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_offsets = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_offsets, sizeof(IDX_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_startidces = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_startidces, sizeof(IDX_T), dn->numlinesout_l , stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_lens = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_lens, sizeof(IDX_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_sources = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_sources, sizeof(IDX_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_targets = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_targets, sizeof(IDX_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	/* MPI Revisions */
	dn->sourceidx_l = calloc(dn->numlinesin_l, sizeof(size_t));
	loadsize = fread(dn->sourceidx_l, sizeof(size_t), dn->numlinesin_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->out_unsorted_blocklens = calloc(dn->commsize, sizeof(unsigned int));
	loadsize = fread(dn->out_unsorted_blocklens, sizeof(unsigned int), dn->commsize, stream);
	if (loadsize != dn->commsize) { printf("Failed to load delay network.\n"); exit(-1); }

	for (int i=0; i<dn->commsize; i++) {
		dn->out_unsorted_blocks[i] = calloc(dn->out_unsorted_blocklens[i], sizeof(FLOAT_T));
		loadsize = fread(dn->out_unsorted_blocks[i], sizeof(FLOAT_T),
							dn->out_unsorted_blocklens[i], stream); 
		if (loadsize != dn->out_unsorted_blocklens[i]) { 
			printf("Failed to load delay network.\n");
			exit(-1);
		}
	}

	dn->out_unsorted_l = calloc(dn->numlinesin_l, sizeof(FLOAT_T));
	loadsize = fread(dn->out_unsorted_l, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	if (loadsize != dn->numlinesin_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->outputs_l = calloc(dn->numlinesin_l, sizeof(FLOAT_T));
	loadsize = fread(dn->outputs_l, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	if (loadsize != dn->numlinesin_l) { printf("Failed to load delay network.\n"); exit(-1); }

	for (int i=0; i<dn->commsize; i++) {
		dn->block_idcs[i] = calloc(dn->out_unsorted_blocklens[i], sizeof(size_t));
		loadsize = fread(dn->block_idcs, sizeof(size_t),
							dn->out_unsorted_blocklens[i], stream);
		if (loadsize != dn->out_unsorted_blocklens[i]) { 
			printf("Failed to load delay network.\n");
			exit(-1);
		}
	}

	return dn;
}


void dn_mpi_save(dn_mpi_delaynet *dn, FILE *stream) {
	fwrite(&dn->num_nodes_g, sizeof(IDX_T), 1, stream);
	fwrite(&dn->num_nodes_l, sizeof(IDX_T), 1, stream);
	fwrite(&dn->nodeoffset, sizeof(IDX_T), 1, stream);
	fwrite(&dn->numlinesout_l, sizeof(IDX_T), 1, stream);
	fwrite(&dn->numlinesin_l, sizeof(IDX_T), 1, stream);
	fwrite(&dn->numlines_g, sizeof(IDX_T), 1, stream);
	fwrite(&dn->lineoffset_in, sizeof(IDX_T), 1, stream);
	fwrite(&dn->lineoffset_out, sizeof(IDX_T), 1, stream);
	fwrite(&dn->buf_len, sizeof(IDX_T), 1, stream);
	fwrite(&dn->commrank, sizeof(int), 1, stream);
	fwrite(&dn->commsize, sizeof(int), 1, stream);

	/*
	fwrite(dn->delaybuf, sizeof(FLOAT_T), dn->buf_len, stream);
	fwrite(dn->inputs, sizeof(FLOAT_T), dn->numlinesout_l, stream);
	fwrite(dn->outputs, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	fwrite(dn->outputs_unsorted, sizeof(FLOAT_T), dn->numlines_g, stream);
	*/

	fwrite(dn->nodes, sizeof(dn_mpi_node), dn->num_nodes_l, stream);
	fwrite(dn->destidx_g, sizeof(IDX_T), dn->numlines_g, stream);
	fwrite(dn->sourceidx_g, sizeof(IDX_T), dn->numlines_g, stream);
	fwrite(dn->del_offsets, sizeof(IDX_T), dn->numlinesout_l, stream);
	fwrite(dn->del_startidces, sizeof(IDX_T), dn->numlinesout_l , stream);
	fwrite(dn->del_lens, sizeof(IDX_T), dn->numlinesout_l, stream);
	fwrite(dn->del_sources, sizeof(IDX_T), dn->numlinesout_l, stream);
	fwrite(dn->del_targets, sizeof(IDX_T), dn->numlinesout_l, stream);

	/* MPI Revisions */
	fwrite(dn->sourceidx_l, sizeof(size_t), dn->numlinesin_l, stream);
	fwrite(dn->out_unsorted_blocklens, sizeof(unsigned int), dn->commsize, stream);
	for (int i=0; i<dn->commsize; i++) {
		fwrite(dn->out_unsorted_blocks[i], sizeof(FLOAT_T),
				dn->out_unsorted_blocklens[i], stream); 
	}
	fwrite(dn->out_unsorted_l, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	fwrite(dn->outputs_l, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	for (int i=0; i<dn->commsize; i++) {
		fwrite(dn->block_idcs, sizeof(size_t),
				dn->out_unsorted_blocklens[i], stream);
	}
}


dn_mpi_delaynet *dn_mpi_load(FILE *stream) {
	size_t loadsize;
	dn_mpi_delaynet *dn;
	dn = malloc(sizeof(dn_mpi_delaynet));

	/* load constants */
	loadsize = fread(&dn->num_nodes_g, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->num_nodes_l, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->nodeoffset, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->numlinesout_l, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->numlinesin_l, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->numlines_g, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->lineoffset_in, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->lineoffset_out, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->buf_len, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->commrank, sizeof(int), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->commsize, sizeof(int), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }

	/* load big chunks */

	/*
	dn->delaybuf = malloc(sizeof(FLOAT_T)*dn->buf_len);
	loadsize = fread(dn->delaybuf, sizeof(FLOAT_T), dn->buf_len, stream);
	if (loadsize != dn->buf_len) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->inputs = malloc(sizeof(FLOAT_T)*dn->numlinesin_l);
	loadsize = fread(dn->inputs, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	if (loadsize != dn->numlinesin_l) { printf("Failed to load delay network.\n"); exit(-1); }
	
	dn->outputs = malloc(sizeof(FLOAT_T)*dn->numlinesout_l);
	loadsize = fread(dn->outputs, sizeof(FLOAT_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->outputs_unsorted = malloc(sizeof(FLOAT_T)*dn->numlines_g);
	loadsize = fread(dn->outputs_unsorted, sizeof(FLOAT_T), dn->numlines_g, stream);
	if (loadsize != dn->numlines_g) { printf("Failed to load delay network.\n"); exit(-1); }
	*/

	dn->nodes = malloc(sizeof(dn_mpi_node)*dn->num_nodes_l);
	loadsize = fread(dn->nodes, sizeof(dn_mpi_node), dn->num_nodes_l, stream);
	if (loadsize != dn->num_nodes_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->destidx_g = malloc(sizeof(IDX_T)*dn->numlines_g);
	loadsize = fread(dn->destidx_g, sizeof(IDX_T), dn->numlines_g, stream);
	if (loadsize != dn->numlines_g) { printf("Failed to load delay network.\n"); exit(-1); }
	
	dn->sourceidx_g = malloc(sizeof(IDX_T)*dn->numlines_g);
	loadsize = fread(dn->sourceidx_g, sizeof(IDX_T), dn->numlines_g, stream);
	if (loadsize != dn->numlines_g) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_offsets = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_offsets, sizeof(IDX_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_startidces = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_startidces, sizeof(IDX_T), dn->numlinesout_l , stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_lens = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_lens, sizeof(IDX_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_sources = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_sources, sizeof(IDX_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_targets = malloc(sizeof(IDX_T)*dn->numlinesout_l);
	loadsize = fread(dn->del_targets, sizeof(IDX_T), dn->numlinesout_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	/* MPI Revisions */
	dn->sourceidx_l = calloc(dn->numlinesin_l, sizeof(size_t));
	loadsize = fread(dn->sourceidx_l, sizeof(size_t), dn->numlinesin_l, stream);
	if (loadsize != dn->numlinesout_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->out_unsorted_blocklens = calloc(dn->commsize, sizeof(unsigned int));
	loadsize = fread(dn->out_unsorted_blocklens, sizeof(unsigned int), dn->commsize, stream);
	if (loadsize != dn->commsize) { printf("Failed to load delay network.\n"); exit(-1); }

	for (int i=0; i<dn->commsize; i++) {
		dn->out_unsorted_blocks[i] = calloc(dn->out_unsorted_blocklens[i], sizeof(FLOAT_T));
		loadsize = fread(dn->out_unsorted_blocks[i], sizeof(FLOAT_T),
							dn->out_unsorted_blocklens[i], stream); 
		if (loadsize != dn->out_unsorted_blocklens[i]) { 
			printf("Failed to load delay network.\n");
			exit(-1);
		}
	}

	dn->out_unsorted_l = calloc(dn->numlinesin_l, sizeof(FLOAT_T));
	loadsize = fread(dn->out_unsorted_l, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	if (loadsize != dn->numlinesin_l) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->outputs_l = calloc(dn->numlinesin_l, sizeof(FLOAT_T));
	loadsize = fread(dn->outputs_l, sizeof(FLOAT_T), dn->numlinesin_l, stream);
	if (loadsize != dn->numlinesin_l) { printf("Failed to load delay network.\n"); exit(-1); }

	for (int i=0; i<dn->commsize; i++) {
		dn->block_idcs[i] = calloc(dn->out_unsorted_blocklens[i], sizeof(size_t));
		loadsize = fread(dn->block_idcs, sizeof(size_t),
							dn->out_unsorted_blocklens[i], stream);
		if (loadsize != dn->out_unsorted_blocklens[i]) { 
			printf("Failed to load delay network.\n");
			exit(-1);
		}
	}

	/* allocate big chunks -- no reading*/
	dn->delaybuf = calloc(dn->buf_len, sizeof(FLOAT_T));
	dn->inputs = calloc(dn->numlinesout_l, sizeof(FLOAT_T));
	dn->outputs = calloc(dn->numlinesin_l, sizeof(FLOAT_T));
	dn->outputs_unsorted = calloc(dn->numlines_g, sizeof(FLOAT_T));

	return dn;
}
