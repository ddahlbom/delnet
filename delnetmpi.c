#include <stdlib.h>
#include <stdio.h> 	//for profiling/debugging
#include <time.h> 	//for profiling/debugging
//#include <mpi.h> 
#include "/usr/include/mpich/mpi.h"

#include "delnetmpi.h"


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
dn_mpi_vec_float dn_mpi_getinputvec(dn_mpi_delaynet *dn) {
	dn_mpi_vec_float inputs;	
	inputs.data = malloc(sizeof(FLOAT_T)*dn->num_delays);
	inputs.n = dn->num_delays;
	for (int i=0; i<dn->num_delays; i++) {
		inputs.data[i] = dn->inputs[i];
	}
	return inputs;
}

/* get inputs to neurons (outputs of delaynet)... */
FLOAT_T *dn_mpi_getinputaddress(IDX_T idx, dn_mpi_delaynet *dn) {
	return &dn->outputs[dn->nodes[idx].idx_outbuf];
}


void dn_mpi_advance(dn_mpi_delaynet *dn)
{
	IDX_T k;

	/* load network inputs into buffers */
	for(k=0; k < dn->num_delays; k++) {
		dn->delaybuf[dn->del_startidces[k]+dn->del_offsets[k]] = dn->inputs[k];
	}

	/* advance the buffers */
	for(k=0; k < dn->num_delays; k++) {
		dn->del_offsets[k] = (dn->del_offsets[k] + 1) % dn->del_lens[k];
	}

	/* pull network outputs from buffers */
	for (k=0; k < dn->num_delays; k++) {
		dn->outputs_unsorted[k] =
				dn->delaybuf[dn->del_startidces[k]+
								dn->del_offsets[k]];
	}

	/* sort output for access */
	for (k=0; k< dn->num_delays; k++) {
		dn->outputs[k] = dn->outputs_unsorted[ dn->sourceidx[k] ];
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
											int commsize, int commrank)
{
	unsigned int i, j, delcount, startidx;
	unsigned int deltot, numlines;
	dn_mpi_delaynet *dn;
	dn_mpi_list_uint **nodes_in;

	dn = malloc(sizeof(dn_mpi_delaynet));
	nodes_in = malloc(sizeof(dn_mpi_list_uint)*n);
	for (i=0; i<n; i++)
		nodes_in[i] = dn_mpi_list_uint_init();

	deltot = 0;
	numlines = 0;
	for (i=0; i<n*n; i++) {
		deltot += g[i];
		numlines += g[i] != 0 ? 1 : 0;
	}
	dn->num_delays = numlines;
	dn->buf_len = deltot;
	dn->num_nodes = n;

	dn->delaybuf = calloc(deltot, sizeof(FLOAT_T));
	dn->inputs = calloc(numlines, sizeof(FLOAT_T));
	dn->outputs = calloc(numlines, sizeof(FLOAT_T));
	dn->outputs_unsorted = calloc(numlines, sizeof(FLOAT_T));

	dn->del_offsets = malloc(sizeof(IDX_T)*numlines);
	dn->del_startidces = malloc(sizeof(IDX_T)*numlines);
	dn->del_lens = malloc(sizeof(IDX_T)*numlines);
	dn->del_sources = malloc(sizeof(IDX_T)*numlines);
	dn->del_targets = malloc(sizeof(IDX_T)*numlines);
	dn->nodes = malloc(sizeof(dn_mpi_node)*n);

	/* init nodes */
	for (i=0; i<n; i++) {
		dn->nodes[i].idx_outbuf = 0;
		dn->nodes[i].num_in = 0;
		dn->nodes[i].idx_inbuf = 0;
		dn->nodes[i].num_out = 0;
	}

	/* work through graph, allocate delay lines */
	delcount = 0;
	startidx = 0;
	for (i = 0; i<n; i++)
	for (j = 0; j<n; j++) {
		if (g[i*n + j] != 0) {
			dn_mpi_list_uint_push(nodes_in[j], i);

			dn->del_offsets[delcount] = 0;
			dn->del_startidces[delcount] = startidx;
			dn->del_lens[delcount] = g[i*n+j];
			dn->del_sources[delcount] = i;
			dn->del_targets[delcount] = j;

			dn->nodes[i].num_out += 1;

			startidx += g[i*n +j];
			delcount += 1;
		}
	}
	
	/* work out rest of index arithmetic */
	unsigned int *num_outputs, *in_base_idcs;
	num_outputs = calloc(n, sizeof(unsigned int));
	in_base_idcs = calloc(n, sizeof(unsigned int));
	for (i=0; i<n; i++) {
		num_outputs[i] = dn->nodes[i].num_out;
		for (j=0; j<i; j++)
			in_base_idcs[i] += num_outputs[j]; 	// check logic here
		//in_base_idcs[i] = i == 0 ? 0 : in_base_idcs[i-1] + num_outputs[i];
	}

	unsigned int idx = 0;
	for (i=0; i<n; i++) {
		dn->nodes[i].num_in = nodes_in[i]->count;
		dn->nodes[i].idx_outbuf = idx;
		idx += dn->nodes[i].num_in;
		dn->nodes[i].idx_inbuf = in_base_idcs[i];
	}

	unsigned int *num_inputs, *out_base_idcs, *out_counts, *inverseidces;
	num_inputs = calloc(n, sizeof(unsigned int));
	out_base_idcs = calloc(n, sizeof(unsigned int));
	out_counts = calloc(n, sizeof(unsigned int));
	for (i=0; i<n; i++) {
		num_inputs[i] = dn->nodes[i].num_in;
		for (j=0; j<i; j++)
			out_base_idcs[i] += num_inputs[j]; // check logic here
		//out_base_idcs[i] = i == 0 ? 0 : in_base_idcs[i-1] + num_inputs[i];
	}

	inverseidces = calloc(numlines, sizeof(unsigned int));
	for (i=0; i < numlines; i++) {
		inverseidces[i] = out_base_idcs[dn->del_targets[i]] + 
						  out_counts[dn->del_targets[i]];
		out_counts[dn->del_targets[i]] += 1;
	}
	dn->destidx = inverseidces;
	dn->sourceidx = malloc(sizeof(unsigned int)*numlines);
	for (i=0; i<numlines; i++)
		dn->sourceidx[dn->destidx[i]] = i;

	/* Clean up */
	for (i=0; i<n; i++)
		dn_mpi_list_uint_free(nodes_in[i]);
	free(nodes_in);
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
	free(dn->destidx);
	free(dn->sourceidx);
	free(dn->delaybuf);
	free(dn->nodes);
	free(dn);
}


void dn_mpi_savecheckpt(dn_mpi_delaynet *dn, FILE *stream) {
	fwrite(&dn->num_nodes, sizeof(IDX_T), 1, stream);
	fwrite(&dn->num_delays, sizeof(IDX_T), 1, stream);
	fwrite(&dn->buf_len, sizeof(IDX_T), 1, stream);

	fwrite(dn->delaybuf, sizeof(FLOAT_T), dn->buf_len, stream);
	fwrite(dn->inputs, sizeof(FLOAT_T), dn->num_delays, stream);
	fwrite(dn->outputs, sizeof(FLOAT_T), dn->num_delays, stream);

	fwrite(dn->nodes, sizeof(dn_mpi_node), dn->num_nodes, stream);

	fwrite(dn->destidx, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->sourceidx, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->del_offsets, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->del_startidces, sizeof(IDX_T), dn->num_delays , stream);
	fwrite(dn->del_lens, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->del_sources, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->del_targets, sizeof(IDX_T), dn->num_delays, stream);
}


dn_mpi_delaynet *dn_mpi_loadcheckpt(FILE *stream) {
	size_t loadsize;
	dn_mpi_delaynet *dn;
	dn = malloc(sizeof(dn_mpi_delaynet));

	/* load constants */
	loadsize = fread(&dn->num_nodes, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->num_delays, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->buf_len, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }

	/* load big chunks */
	dn->delaybuf = malloc(sizeof(FLOAT_T)*dn->buf_len);
	loadsize = fread(dn->delaybuf, sizeof(FLOAT_T), dn->buf_len, stream);
	if (loadsize != dn->buf_len) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->inputs = malloc(sizeof(FLOAT_T)*dn->num_delays);
	loadsize = fread(dn->inputs, sizeof(FLOAT_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->outputs = malloc(sizeof(FLOAT_T)*dn->num_delays);
	loadsize = fread(dn->outputs, sizeof(FLOAT_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->nodes = malloc(sizeof(dn_mpi_node)*dn->num_nodes);
	loadsize = fread(dn->nodes, sizeof(dn_mpi_node), dn->num_nodes, stream);
	if (loadsize != dn->num_nodes) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->destidx = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->destidx, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->sourceidx = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->sourceidx, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }
	
	dn->del_offsets = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_offsets, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_startidces = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_startidces, sizeof(IDX_T), dn->num_delays , stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_lens = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_lens, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_sources = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_sources, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_targets = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_targets, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	return dn;
}


void dn_mpi_save(dn_mpi_delaynet *dn, FILE *stream) {
	fwrite(&dn->num_nodes, sizeof(IDX_T), 1, stream);
	fwrite(&dn->num_delays, sizeof(IDX_T), 1, stream);
	fwrite(&dn->buf_len, sizeof(IDX_T), 1, stream);

	fwrite(dn->nodes, sizeof(dn_mpi_node), dn->num_nodes, stream);

	fwrite(dn->destidx, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->sourceidx, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->del_offsets, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->del_startidces, sizeof(IDX_T), dn->num_delays , stream);
	fwrite(dn->del_lens, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->del_sources, sizeof(IDX_T), dn->num_delays, stream);
	fwrite(dn->del_targets, sizeof(IDX_T), dn->num_delays, stream);
}


dn_mpi_delaynet *dn_mpi_load(FILE *stream) {
	size_t loadsize;
	dn_mpi_delaynet *dn;
	dn = malloc(sizeof(dn_mpi_delaynet));

	/* load constants */
	loadsize = fread(&dn->num_nodes, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->num_delays, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }
	loadsize = fread(&dn->buf_len, sizeof(IDX_T), 1, stream);
	if (loadsize != 1) { printf("Failed to load delay network.\n"); exit(-1); }

	/* load saved chunks */
	dn->nodes = malloc(sizeof(dn_mpi_node)*dn->num_nodes);
	loadsize = fread(dn->nodes, sizeof(dn_mpi_node), dn->num_nodes, stream);
	if (loadsize != dn->num_nodes) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->destidx = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->destidx, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->sourceidx = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->sourceidx, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }
	
	dn->del_offsets = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_offsets, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_startidces = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_startidces, sizeof(IDX_T), dn->num_delays , stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_lens = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_lens, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_sources = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_sources, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	dn->del_targets = malloc(sizeof(IDX_T)*dn->num_delays);
	loadsize = fread(dn->del_targets, sizeof(IDX_T), dn->num_delays, stream);
	if (loadsize != dn->num_delays) { printf("Failed to load delay network.\n"); exit(-1); }

	/* allocate big chunks -- no reading*/
	dn->delaybuf = calloc(dn->buf_len, sizeof(FLOAT_T));
	dn->inputs = calloc(dn->num_delays, sizeof(FLOAT_T));
	dn->outputs = calloc(dn->num_delays, sizeof(FLOAT_T));

	return dn;
}
