#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>


#include "delnet.h"


/*************************************************************
 *  Macros
 *************************************************************/
//#define SPIKE_BLOCK_SIZE 32768
#define SPIKE_BLOCK_SIZE 8192
#define TPB 256


/*************************************************************
 *  Globals
 *************************************************************/
FLOAT_T g_v_default = -65.0;
FLOAT_T g_u_default = -13.0;

FLOAT_T g_a_exc  = 0.02;
FLOAT_T g_d_exc  = 8.0;
FLOAT_T g_w_exc  = 6.0;

FLOAT_T g_a_inh  = 0.1;
FLOAT_T g_d_inh  = 2.0;
FLOAT_T g_w_inh = -5.0;


/*************************************************************
 *  Structs
 *************************************************************/
typedef struct neuron_s {
	FLOAT_T v;
	FLOAT_T u;
	FLOAT_T a;
	FLOAT_T d;
} neuron;

typedef struct spike_s {
	int neuron;
	FLOAT_T time;
} spike;

typedef struct spikeblock_s {
	long max_spikes;
	long num_spikes;
	spike *spikes;
	struct spikeblock_s *next;
} spikeblock;

typedef struct spikerecord_s {
	spikeblock *head;	
} spikerecord;


/*************************************************************
 *  Functions
 *************************************************************/
__global__ void synapse_trace_kernel_cuda(IDX_T n,
										  IDX_T *offsets,
										  IDX_T *nums_in,
										  FLOAT_T *spike_pre,
										  FLOAT_T *trace_pre, 
										  FLOAT_T *neuroninputs,
										  FLOAT_T dt,
										  FLOAT_T tau_pre) 
{
	unsigned int i, j;
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int stride = blockDim.x * gridDim.x; 

	for (i=index; i < n; i += stride) {
		for (j=0; j < nums_in[i]; j++) {
			spike_pre[offsets[i]+j] = neuroninputs[offsets[i]+j];
			trace_pre[offsets[i]+j] =
				trace_pre[offsets[i]+j]*(1.0 - (dt/tau_pre)) +
							  spike_pre[offsets[i]+j];
		}
	}
}

void synapse_trace_update_cuda(IDX_T n_nodes,
							   IDX_T n_inputs,
							   IDX_T *offsets,
							   IDX_T *nums_in,
							   FLOAT_T *spike_pre,
							   FLOAT_T *trace_pre, 
							   FLOAT_T *neuroninputs,
							   FLOAT_T dt,
							   FLOAT_T tau_pre) 
{
	unsigned int numblocks = (n_nodes + TPB - 1) / TPB;		
	IDX_T *d_offsets=0, *d_nums_in=0;
	FLOAT_T *d_spike_pre=0, *d_trace_pre=0, *d_neuroninputs=0;

	/* move data to GPU */
	cudaMemcpy(d_offsets, offsets, n_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nums_in, nums_in, n_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_spike_pre, spike_pre, n_inputs, cudaMemcpyHostToDevice);
	cudaMemcpy(d_trace_pre, trace_pre, n_inputs, cudaMemcpyHostToDevice);
	cudaMemcpy(d_neuroninputs, neuroninputs, n_inputs, cudaMemcpyHostToDevice);

	synapse_trace_kernel_cuda<<<numblocks, TPB>>>(n_nodes,
												  d_offsets,
												  nums_in,
												  spike_pre,
												  trace_pre,
												  neuroninputs,
												  dt,
												  tau_pre);

	/* move data back from GPU to main memory */
	cudaMemcpy(offsets, d_offsets, n_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(nums_in, d_nums_in, n_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(spike_pre, d_spike_pre, n_inputs, cudaMemcpyDeviceToHost);
	cudaMemcpy(trace_pre, d_trace_pre, n_inputs, cudaMemcpyDeviceToHost);
	cudaMemcpy(neuroninputs, d_neuroninputs, n_inputs, cudaMemcpyDeviceToHost);
}

static inline void synapse_trace_update(IDX_T n,
										IDX_T *offsets,
										IDX_T *nums_in,
										FLOAT_T *spike_pre,
										FLOAT_T *trace_pre, 
										FLOAT_T *neuroninputs,
										FLOAT_T dt,
										FLOAT_T tau_pre) 
{
	IDX_T k,j;

	for (k=0; k<n; k++) {
		for (j=0; j < nums_in[k]; j++) {
			spike_pre[offsets[k]+j] = neuroninputs[offsets[k]+j];
			trace_pre[offsets[k]+j] =
				trace_pre[offsets[k]+j]*(1.0 - (dt/tau_pre)) +
							  spike_pre[offsets[k]+j];
		}
	}
}



static inline void synapse_strength_update(IDX_T n_exc,
										   IDX_T *exc_offset,
										   IDX_T *offsets,
										   IEX_T *nums_in,
										   trace_pre,
										   trace_post,
										   spike_pre,
										   spike_post,
										   synapses,
										   dt,
										   a_pre,
										   a_post,
										   synmax)
{
}

double dd_sum_double(double *vals, size_t n) {
	double sum = 0.0;
	for (size_t k=0; k<n; k++) 
		sum += vals[k];
	return sum;
}

double dd_avg_double(double *vals, size_t n) {
	double sum = dd_sum_double(vals, n);
	return sum / ((double) n);
}

spikerecord *sr_init()
{
	spikerecord *rec;
	rec = (spikerecord *) malloc(sizeof(spikerecord));
	rec->head = (spikeblock *) malloc(sizeof(spikeblock));
	rec->head->max_spikes = SPIKE_BLOCK_SIZE;
	rec->head->num_spikes = 0;
	rec->head->spikes = (spike *) malloc(sizeof(spike)*SPIKE_BLOCK_SIZE);
	rec->head->next = 0;

	return rec;
}

void sr_save_spike(spikerecord *sr, int neuron, FLOAT_T time)
{
	if (sr->head->num_spikes < sr->head->max_spikes) {
		sr->head->spikes[sr->head->num_spikes].neuron = neuron;
		sr->head->spikes[sr->head->num_spikes].time = time;
		sr->head->num_spikes += 1;
	}
	else {
		/* allocate new spike block and saves spike */
		spikeblock *newest = (spikeblock *) malloc(sizeof(spikeblock));
		newest->max_spikes = SPIKE_BLOCK_SIZE;
		newest->num_spikes = 0;
		newest->spikes = (spike *) malloc(sizeof(spike)*SPIKE_BLOCK_SIZE);
		newest->next = sr->head;
		sr->head = newest;
		sr->head->spikes[sr->head->num_spikes].neuron = neuron;
		sr->head->spikes[sr->head->num_spikes].time = time;
		sr->head->num_spikes += 1;
	}
}


/*
 * Revise this later so that spikes are in order (they are in order
 * by block, but blocks are reversed)
 */
spike *sr_spike_summary(spikerecord *sr)
{
	/* Calculate total number of spikes and allocate */
	long num_spikes = 0;
	spike *spikes_all;
	spikeblock *curblock = sr->head;
	while (curblock != 0) {
		num_spikes += curblock->num_spikes;
		curblock = curblock->next;
	}

	spikes_all = (spike *) malloc(sizeof(spike)*num_spikes);

	curblock = sr->head;
	long idx = 0;
	while (curblock != 0) {
		for (int i=0; i < curblock->num_spikes; i++) {
			spikes_all[idx] = curblock->spikes[i];
			idx += 1;
		}
		curblock = curblock->next;
	}
	return spikes_all;
}

void sr_free(spikerecord *sr)
{
	spikeblock *curblock = sr->head;
	spikeblock *newest;
	while (curblock != 0) {
		free(curblock->spikes);
		newest = curblock->next;
		free(curblock);
		curblock = newest;
	}
	free(sr);
}

/*
 * -------------------- Util Functions --------------------
 */

dn_list_uint *dn_list_uint_init() {
	dn_list_uint *newlist;
	newlist = (dn_list_uint *)malloc(sizeof(dn_list_uint));
	newlist->count = 0;
	newlist->head = NULL; 

	return newlist;
}

void dn_list_uint_push(dn_list_uint *l, unsigned int val) {
	dn_listnode_uint *newnode;
	newnode = (dn_listnode_uint *)malloc(sizeof(dn_listnode_uint));
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
	output.data = (FLOAT_T *)malloc(sizeof(FLOAT_T) * n);

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
	output = (char *) malloc(sizeof(char)*(input.n+1));
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
	inputs.data = (FLOAT_T *) malloc(sizeof(FLOAT_T)*dn->num_delays);
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
	delmat = (unsigned int *) malloc(sizeof(unsigned int)*n*n);

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

	dn = (dn_delaynet *) malloc(sizeof(dn_delaynet));
	nodes_in = (dn_list_uint **) malloc(sizeof(dn_list_uint *)*n);
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

	dn->delaybuf = (FLOAT_T *) calloc(deltot, sizeof(FLOAT_T));
	dn->inputs   = (FLOAT_T *) calloc(numlines, sizeof(FLOAT_T));
	dn->outputs  = (FLOAT_T *) calloc(numlines, sizeof(FLOAT_T));

	dn->del_offsets 	= (IDX_T *) malloc(sizeof(IDX_T)*numlines);
	dn->del_startidces 	= (IDX_T *) malloc(sizeof(IDX_T)*numlines);
	dn->del_lens 		= (IDX_T *) malloc(sizeof(IDX_T)*numlines);
	dn->del_sources 	= (IDX_T *) malloc(sizeof(IDX_T)*numlines);
	dn->del_targets 	= (IDX_T *) malloc(sizeof(IDX_T)*numlines);
	dn->nodes 			= (dn_node *) malloc(sizeof(dn_node)*n);

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
	num_outputs  = (unsigned int *) calloc(n, sizeof(unsigned int));
	in_base_idcs = (unsigned int *) calloc(n, sizeof(unsigned int));
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
	num_inputs 		= (unsigned int *) calloc(n, sizeof(unsigned int));
	out_base_idcs 	= (unsigned int *) calloc(n, sizeof(unsigned int));
	out_counts 		= (unsigned int *) calloc(n, sizeof(unsigned int));
	for (i=0; i<n; i++) {
		num_inputs[i] = dn->nodes[i].num_in;
		for (j=0; j<i; j++)
			out_base_idcs[i] += num_inputs[j]; // check logic here
		//out_base_idcs[i] = i == 0 ? 0 : in_base_idcs[i-1] + num_inputs[i];
	}

	inverseidces = (unsigned int *) calloc(numlines, sizeof(unsigned int));
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



/*************************************************************
 *  Main
 *************************************************************/

/**
 * @brief Simulation with delnet library
 *
 */
int main(int argc, char *argv[])
{
	FLOAT_T fs, dur, dt, t;
	FLOAT_T tau_pre, tau_post, a_pre, a_post, synbump, synmax;
	unsigned int i, j, k, numsteps;
	unsigned int n, n_exc, n_inh;
	unsigned int *g;
	unsigned long int numspikes = 0;
	float p_contact;
	dn_delaynet *dn;
	spikerecord *sr = sr_init();
	clock_t t_start, t_finish;
	double *gettinginputs, *updatingsyntraces, *updatingneurons, *spikechecking,
			*updatingneutraces, *updatingsynstrengths, *pushingoutput,
			*advancingbuffer;

	if (argc != 2)
		n = 1000;
	else
		n = atoi(argv[1]);


	/* trial parameters */
	fs = 1000.0;
	dur = 2.0;
	p_contact = 0.1;
	//n = 2000;
	tau_pre = 0.02;
	tau_post = 0.02;
	a_pre = 0.12;
	a_post = 0.1;
	synbump = 0.00000;
	synmax = 10.0;



	/* derived parameters */
	n_exc = n*0.8;
	n_inh = n*0.2;
	n = n_exc + n_inh;  // in case of rounding issue
	dt = 1.0/fs;
	numsteps = dur/dt;

	/* print trial parameters */
	printf("Sampling Frequency: \t%f\n", fs);
	printf("Duration: \t\t%f\n", dur);
	printf("Number of nodes: \t%d\n", n);
	printf(" 	Excitatory: \t%d\n", n_exc);
	printf(" 	Inhibitory: \t%d\n", n_inh);
	printf("Probability of contact:\t%f\n", p_contact);
	printf("tau_pre:\t\t%f\n", tau_pre);
	printf("A_pre:\t\t\t%f\n", a_pre);
	printf("tau_post:\t\t%f\n", tau_pre);
	printf("A_post:\t\t\t%f\n", a_pre);
	printf("----------------------------------------\n");

	/* set up graph */
	g = dn_blobgraph(n, p_contact, 20);
	for (i=n_exc; i<n; i++) 			// only last 200 rows
	for (j=0; j<n; j++) { 				
		g[i*n+j] = g[i*n+j] != 0 ? 1 : 0; 	// 1 ms delay for inh
	}

	/* generate delay network */
	dn = dn_delnetfromgraph(g, n);

	/* initialize neuron and synapse state  */
	neuron *neurons 	= (neuron *) malloc(sizeof(neuron)*n);
	FLOAT_T *trace_post = (FLOAT_T *) calloc(n_exc, sizeof(FLOAT_T));
	FLOAT_T *spike_post = (FLOAT_T *) calloc(n_exc, sizeof(FLOAT_T));
	IDX_T *offsets 		= (IDX_T *) malloc(sizeof(IDX_T)*n);
	FLOAT_T *trace_pre; 	// pack this
	FLOAT_T *spike_pre; 	// and following
	FLOAT_T *synapses; 		// for speed?

	unsigned long cum_in = 0, exc_offset;
	for (i=0; i<n_exc; i++) {
		neurons[i].v = g_v_default;
		neurons[i].u = g_u_default;
		neurons[i].a = g_a_exc;
		neurons[i].d = g_d_exc;
		offsets[i] = cum_in;
		cum_in += dn->nodes[i].num_in;
	}
	exc_offset = cum_in;

	for (i=n_exc; i<n; i++) {
		neurons[i].v = g_v_default;
		neurons[i].u = g_u_default;
		neurons[i].a = g_a_inh;
		neurons[i].d = g_d_inh;
		offsets[i] = cum_in;
		cum_in += dn->nodes[i].num_in;
	}

	trace_pre = (FLOAT_T *) calloc(exc_offset, sizeof(FLOAT_T));
	spike_pre = (FLOAT_T *) calloc(exc_offset, sizeof(FLOAT_T));
	synapses  = (FLOAT_T *) calloc(cum_in, sizeof(FLOAT_T));
	for (i=0; i<n_exc; i++)
		synapses[i] = g_w_exc;
	for (; i<n; i++)
		synapses[i] = g_w_inh;


	/* for profiling */	
	gettinginputs 		 = (double *) malloc(sizeof(double)*numsteps);
	updatingsyntraces 	 = (double *) malloc(sizeof(double)*numsteps);
	updatingneurons 	 = (double *) malloc(sizeof(double)*numsteps);
	spikechecking 		 = (double *) malloc(sizeof(double)*numsteps);
	updatingneutraces 	 = (double *) malloc(sizeof(double)*numsteps);
	updatingsynstrengths = (double *) malloc(sizeof(double)*numsteps);
	pushingoutput 		 = (double *) malloc(sizeof(double)*numsteps);
	advancingbuffer 	 = (double *) malloc(sizeof(double)*numsteps);

	/* intermediate variables for simulation -- clean these up later*/
	FLOAT_T *neuroninputs, *invals, *outvals;
	IDX_T *nums_in;
	invals  = (FLOAT_T *) calloc(n, sizeof(FLOAT_T));
	outvals = (FLOAT_T *) calloc(n, sizeof(FLOAT_T));
	nums_in = (IDX_T *) calloc(n, sizeof(IDX_T));
	for (i=0; i<n; i++) 
		nums_in[i] = dn->nodes[i].num_in; 	// see if helps speed


	/* start simulation */
	for (i=0; i<numsteps; i++) {

		/* print updates */
		t = dt*i;
		if (i%1000 == 0)
			printf("Time: %f\n", t);


		/* get inputs to neuron */		
		t_start = clock();
		for (k=0; k<n; k++) {
			neuroninputs = dn_getinputaddress(k, dn);

			/* weighted sum */
			for (j=0; j < nums_in[k]; j++)
				invals[k] += *(neuroninputs+j) * synapses[offsets[k]+j];

			/* added noise */
			if (unirand() < 1.0/n)
				invals[k] += 20.0 * (fs/1000.0);
		}
		t_finish = clock();
		gettinginputs[i] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update synapse traces */
		t_start = clock();
	    synapse_trace_update_cuda(n_exc, 		// number of excitatory neurons
	  						      exc_offset, 	// number of excitatory synapses
	  						      offsets,
	  						      nums_in,
	  						      spike_pre,
	  						      trace_pre, 
	  						      neuroninputs,
	  						      dt,
	  						      tau_pre);
		t_finish = clock();
		updatingsyntraces[i] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update neuron state */
		t_start = clock();
		for (k=0; k<n; k++) {
			neurons[k].v += 500.0 * dt * (( 0.04 * neurons[k].v + 5.0) *
							neurons[k].v + 140.0 - neurons[k].u + invals[k]);
			neurons[k].v += 500.0 * dt * (( 0.04 * neurons[k].v + 5.0) *
							neurons[k].v + 140.0 - neurons[k].u + invals[k]);
			neurons[k].u += 1000.0 * dt * neurons[k].a *
								(0.2 * neurons[k].v - neurons[k].u);
			invals[k] = 0.0;
		}
		t_finish = clock();
		updatingneurons[i] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

		/* check if spiked and calculate output */
		t_start = clock();
		for (k=0; k<n; k++) {
			outvals[k] = 0.0;
			if (neurons[k].v >= 30.0) {
				sr_save_spike(sr, k, t);
				outvals[k] = 1.0;
				neurons[k].v = -65.0;
				neurons[k].u += neurons[k].d;
				numspikes += 1;
			}
		}
		t_finish = clock();
		spikechecking[i] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

		/* update neuron trace */		
		t_start = clock();
		for (k=0; k<n; k++) {
			if (k < n_exc) {
				spike_post[k] = outvals[k];
				trace_post[k] = trace_post[k]*(1.0 - (dt/tau_post)) +
								spike_post[k];
			}
		}
		t_finish = clock();
		updatingneutraces[i] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update synapse strengths */
		t_start = clock();
		synapse_strength_update(n_exc,
							    exc_offset,
								offsets,
								nums_in,
								trace_pre,
								trace_post,
								spike_pre,
								spike_post,
								synapses,
								dt,
								a_pre,
								a_post,
								synmax);
		/*
		for (k=0; k<n_exc; k++) {
			for (j=0; j < nums_in[k]; j++) {
				synapses[offsets[k]+j] = synapses[offsets[k]+j] + synbump +
						dt * (a_post * trace_pre[offsets[k]+j] * spike_post[k] -
							  a_pre * trace_post[k] * spike_pre[offsets[k]+j]);
				synapses[offsets[k]+j] =
					synapses[offsets[k]+j] < 0.0 ? 0.0 : synapses[offsets[k]+j];
				synapses[offsets[k]+j] =
					synapses[offsets[k]+j] > synmax ? synmax : synapses[offsets[k]+j];
				//spike_post[k] = 0.0;
			}
		}
		*/
		t_finish = clock();
		updatingsynstrengths[i] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* push the output into the buffer */
		t_start = clock();
		for (k=0; k<n; k++)
			dn_pushoutput(outvals[k], k, dn);
		t_finish = clock();
		pushingoutput[i] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* advance the buffer */
		t_start = clock();
		dn_advance(dn);
		t_finish = clock();
		advancingbuffer[i] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
	}

	/* performance analysis */
	double cycletime, cumtime = 0.0;

	printf("----------------------------------------\n");

	cycletime = 1000.0*dd_sum_double(gettinginputs, numsteps)/numsteps;
	cumtime += cycletime;
	printf("Getting inputs:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(updatingsyntraces, numsteps)/numsteps;
	cumtime += cycletime;
	printf("Update syntraces:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(updatingneurons, numsteps)/numsteps;
	cumtime += cycletime;
	printf("Update neurons:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(spikechecking, numsteps)/numsteps;
	cumtime += cycletime;
	printf("Check spiked:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(updatingneutraces, numsteps)/numsteps;
	cumtime += cycletime;
	printf("Update neurtrace:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(updatingsynstrengths, numsteps)/numsteps;
	cumtime += cycletime;
	printf("Update syn strength:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(pushingoutput, numsteps)/numsteps;
	cumtime += cycletime;
	printf("Pushing buffer:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(advancingbuffer, numsteps)/numsteps;
	cumtime += cycletime;
	printf("Advancing buffer:\t %f (ms)\n", cycletime);

	printf("Total cycle time:\t %f (ms)\n", cumtime);
	printf("Time per second: \t %f (ms)\n", cumtime*fs);

	/* Save spikes */
	FILE *spike_file;
	spike_file = fopen( "delnetstdp.dat", "w" );
	spike *firings = sr_spike_summary(sr);
	for (i=0; i<numspikes; i++)
		fprintf(spike_file, "%f  %d\n", firings[i].time, firings[i].neuron);
	fclose(spike_file);


	/* Clean up */
	dn_freedelnet(dn);
	sr_free(sr);
	free(g);
	free(firings);
	free(trace_post);
	free(spike_post);
	free(neurons);
	free(trace_pre);
	free(spike_pre);
	free(synapses);
	free(gettinginputs);
	free(updatingsyntraces);
	free(updatingneurons);
	free(spikechecking);
	free(updatingneutraces);
	free(updatingsynstrengths);
	free(pushingoutput);
	free(advancingbuffer);

	return 0;
}
