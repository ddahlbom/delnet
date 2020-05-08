#include <stdlib.h>
#include <stdio.h> 	//for profiling/debugging
#include <time.h> 	//for profiling/debugging
#include <math.h>

#ifdef __amd64__
#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif

#include "delnetmpi.h"

#define DEBUG 0

#define OFFSET 0
#define LENGTH 1
#define DATA 2


/* --------------------	MPI Utils -------------------- */
size_t dn_mpi_maxnode(int rank, int commsize, size_t numpoints)
{
	size_t basesize = floor(numpoints/commsize);
	return rank < (numpoints % commsize) ? basesize + 1 : basesize;
}

size_t dn_mpi_nodeoffset(int rank, int commsize, size_t numpoints)
{
	size_t offset = 0;
	int i=0;
	while (i < rank) {
		offset += dn_mpi_maxnode(rank, commsize, numpoints);
		i++;
	}
	return offset;
}

/*
 * Reconstructs the global outbuffer vector on every rank.
 */
void dn_mpi_syncoutputs(dn_mpi_delaynet *dn)
{
	unsigned int i;
	MPI_Request dataRecvReq[dn->commsize];
	MPI_Request dataSendReq[dn->commsize];

	/* Non-blocking receives for actual data */
	for (i=0; i<dn->commsize; i++) {
		if (i!=dn->commrank) {
			MPI_Irecv(&dn->outputs_unsorted[dn->outblockoffsets[i]],
					  dn->outblocksizes[i],
					  MPI_DOUBLE,
					  i,
					  DATA, 
					  MPI_COMM_WORLD,
					  &dataRecvReq[i]);
		}
	}

	/* Non-blocking sends */
	for (i=0; i<dn->commsize; i++) {
		if (i!=dn->commrank) {
			MPI_Isend(&dn->outputs_unsorted[dn->lineoffset_in],
					  dn->numlinesout_l,
					  MPI_DOUBLE,
					  i,
					  DATA,
					  MPI_COMM_WORLD,
					  &dataSendReq[i]);
		}
	}
	
	/* Wait to receive data information */
	for (i=0; i<dn->commsize; i++) {
		if (i!=dn->commrank) {
			MPI_Wait(&dataRecvReq[i], MPI_STATUS_IGNORE);
		}
	}

	/* Wait for sent data to be received */
	for (i=0; i<dn->commsize; i++) {
		if (i!=dn->commrank) {
			MPI_Wait(&dataSendReq[i], MPI_STATUS_IGNORE);
		}
	}
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


/* get inputs to neurons (outputs of delaynet)... */
FLOAT_T *dn_mpi_getinputaddress(IDX_T idx, dn_mpi_delaynet *dn) {
	return &dn->outputs[dn->nodes[idx].idx_outbuf];
}


void dn_mpi_advance(dn_mpi_delaynet *dn)
{
	IDX_T i, k, k_global;

	/* load network inputs into buffers */
	for(k=0; k < dn->numlinesout_l; k++) {
		dn->delaybuf[dn->del_startidces[k]+dn->del_offsets[k]] = dn->inputs[k];
	}

	/* advance the buffers */
	for(k=0; k < dn->numlinesout_l; k++) {
		dn->del_offsets[k] = (dn->del_offsets[k] + 1) % dn->del_lens[k];
	}

	/* pull network outputs from buffers */
	for (k=0; k < dn->numlinesout_l; k++) {
		k_global = k + dn->lineoffset_in; 	// because still in input buffer order
		dn->outputs_unsorted[k_global] =
				dn->delaybuf[dn->del_startidces[k]+
								dn->del_offsets[k]];
	}

	/* ----- MPI Communication ------ sharing outputs */
	dn_mpi_syncoutputs(dn);

	/* sort output for access */
	for (k=0; k< dn->numlinesin_l; k++) {
		k_global = k + dn->lineoffset_out; 	// <--- confirm this (lineoffset_out?)
		dn->outputs[k] = dn->outputs_unsorted[ dn->sourceidx_g[k_global] ];
	}
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

	// Local backup of original
	IDX_T *del_offsets = malloc(sizeof(IDX_T)*numlines_g);
	IDX_T *del_startidces = malloc(sizeof(IDX_T)*numlines_g);
	IDX_T *del_lens = malloc(sizeof(IDX_T)*numlines_g);
	IDX_T *del_sources = malloc(sizeof(IDX_T)*numlines_g);
	IDX_T *del_targets = malloc(sizeof(IDX_T)*numlines_g);
	dn_mpi_node *nodes = malloc(sizeof(dn_mpi_node)*n);


	/* init nodes -- should've just calloc-ed*/
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
	i1_inbuf = nodes[nodeoffset+num_nodes_l-1].idx_inbuf + nodes[nodeoffset+num_nodes_l-1].num_out;
	i0_outbuf = nodes[nodeoffset].idx_outbuf;
	i1_outbuf = nodes[nodeoffset+num_nodes_l-1].idx_outbuf + nodes[nodeoffset+num_nodes_l-1].num_in;

	dn->lineoffset_in = i0_inbuf;
	dn->lineoffset_out = i0_outbuf;

	// ASSERT!
	if (i1_inbuf-i0_inbuf != numlinesout_l) {
		printf("Process %d -- i1_inbuf-i0_inbuf: %d \t numlinesout_l: %d\n", commrank, i1_inbuf-i0_inbuf, numlinesout_l);
		printf("You f*!@ed up indexing!\n");
		exit(-1);
	}

	if (i1_outbuf-i0_outbuf != numlinesin_l) {
		printf("Process %d -- i1_outbuf-i0_outbuf: %d \t numlinesin_l: %d\n", commrank, i1_outbuf-i0_outbuf, numlinesin_l);
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

	/* ----- MPI ACTIVITY -----  Share blocksize and offset info */
	dn->outblocksizes   = calloc(commsize, sizeof(unsigned int));
	dn->outblockoffsets = calloc(commsize, sizeof(unsigned int));

	MPI_Allgather(&dn->numlinesout_l,
				  1,
				  MPI_UNSIGNED,
				  dn->outblocksizes,
				  1,
				  MPI_UNSIGNED,
				  MPI_COMM_WORLD);

	for (i=0; i<commsize; i++) {
		for (j=0; j<i; j++) {
			dn->outblockoffsets[i] += dn->outblocksizes[j];	
		}
	}


	/* free up data not kept in delnet stucture */
	free(del_offsets);
	free(del_startidces);
	free(del_lens);
	free(del_sources);
	free(del_targets);
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
	free(dn->outblocksizes);
	free(dn->outblockoffsets);
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

	/* allocate big chunks -- no reading*/
	dn->delaybuf = calloc(dn->buf_len, sizeof(FLOAT_T));
	dn->inputs = calloc(dn->numlinesout_l, sizeof(FLOAT_T));
	dn->outputs = calloc(dn->numlinesin_l, sizeof(FLOAT_T));
	dn->outputs_unsorted = calloc(dn->numlines_g, sizeof(FLOAT_T));

	return dn;
}
