#include <stdlib.h>
#include <stdio.h> 	//for profiling/debugging
#include <time.h> 	//for profiling/debugging

#include "delnet.h"


/*
 * -------------------- Util Functions --------------------
 */

dn_list_uint *dn_list_uint_init() {
	dn_list_uint *newlist;
	newlist = malloc(sizeof(dn_list_uint));
	newlist->count = 0;
	newlist->head = NULL; 

	return newlist;
}

void dn_list_uint_push(dn_list_uint *l, unsigned int val) {
	dn_listnode_uint *newnode;
	newnode = malloc(sizeof(dn_listnode_uint));
	newnode->val = val;
	newnode->next = l->head;
	l->head = newnode;
	l->count += 1;
}

unsigned int dn_list_uint_pop(dn_list_uint *l) {
	unsigned int val;
	dn_listnode_uint *temp;

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

void dn_list_uint_free(dn_list_uint *l) {
	while (l->head != NULL) {
		dn_list_uint_pop(l);
	}
	free(l);
}


dn_vec_float dn_orderbuf(IDX_T which, dn_delaynet *dn) {
	IDX_T k, n, idx;
	dn_vec_float output;
	
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

char *dn_vectostr(dn_vec_float input) {
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

void dn_pushoutput(FLOAT_T val, IDX_T idx, dn_delaynet *dn) 
{
	IDX_T i1, i2, k;

	i1 = dn->nodes[idx].idx_io;
	i2 = i1 + dn->nodes[idx].num_out;

	for (k = i1; k < i2; k++)
		dn->inputs[k] = val;
}


/* No getinputs()... would need to return vector */
dn_vec_float dn_getinputvec(dn_delaynet *dn) {
	dn_vec_float inputs;	
	inputs.data = malloc(sizeof(FLOAT_T)*dn->num_delays);
	inputs.n = dn->num_delays;
	for (int i=0; i<dn->num_delays; i++) {
		inputs.data[i] = dn->inputs[i];
	}
	return inputs;
}

/* get inputs to neurons (outputs of delaynet)... */
FLOAT_T *dn_getinputaddress(IDX_T idx, dn_delaynet *dn) {
	return &dn->outputs[dn->nodes[idx].idx_oi];
}


void dn_advance(dn_delaynet *dn)
{
	IDX_T k;

	for(k=0; k < dn->num_delays; k++) {
		dn->delaybuf[dn->del_startidces[k] + dn->del_offsets[k]] =
															dn->inputs[k];
	}

	for(k=0; k < dn->num_delays; k++) {
		dn->del_offsets[k] = (dn->del_offsets[k] + 1) % dn->del_lens[k];
	}

	for (k=0; k < dn->num_delays; k++) {
		dn->outputs[dn->inverseidx[k]] =
				dn->delaybuf[dn->del_startidces[k]+dn->del_offsets[k]];
	}
}


unsigned int *dn_blobgraph(unsigned int n, float p, unsigned int maxdel) {
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


dn_delaynet *dn_delnetfromgraph(unsigned int *g, unsigned int n) {
	unsigned int i, j, delcount, startidx;
	unsigned int deltot, numlines;
	dn_delaynet *dn;
	dn_list_uint **nodes_in;

	dn = malloc(sizeof(dn_delaynet));
	nodes_in = malloc(sizeof(dn_list_uint)*n);
	for (i=0; i<n; i++)
		nodes_in[i] = dn_list_uint_init();

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

	dn->del_offsets = malloc(sizeof(IDX_T)*numlines);
	dn->del_startidces = malloc(sizeof(IDX_T)*numlines);
	dn->del_lens = malloc(sizeof(IDX_T)*numlines);
	dn->del_sources = malloc(sizeof(IDX_T)*numlines);
	dn->del_targets = malloc(sizeof(IDX_T)*numlines);
	dn->nodes = malloc(sizeof(dn_node)*n);

	/* init nodes */
	for (i=0; i<n; i++) {
		dn->nodes[i].idx_oi = 0;
		dn->nodes[i].num_in = 0;
		dn->nodes[i].idx_io = 0;
		dn->nodes[i].num_out = 0;
	}

	/* work through graph, allocate delay lines */
	delcount = 0;
	startidx = 0;
	for (i = 0; i<n; i++)
	for (j = 0; j<n; j++) {
		if (g[i*n + j] != 0) {
			dn_list_uint_push(nodes_in[j], i);

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
		dn->nodes[i].idx_oi = idx;
		idx += dn->nodes[i].num_in;
		dn->nodes[i].idx_io = in_base_idcs[i];
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
	dn->inverseidx = inverseidces;

	/* Clean up */
	for (i=0; i<n; i++)
		dn_list_uint_free(nodes_in[i]);
	free(nodes_in);
	free(num_outputs);
	free(in_base_idcs);
	free(num_inputs);
	free(out_base_idcs);
	free(out_counts);

	return dn;
}

void dn_freedelnet(dn_delaynet *dn) {
	free(dn->del_offsets);
	free(dn->del_startidces);
	free(dn->del_lens);
	free(dn->del_sources);
	free(dn->del_targets);
	free(dn->inputs);
	free(dn->outputs);
	free(dn->inverseidx);
	free(dn->delaybuf);
	free(dn->nodes);
	free(dn);
}
