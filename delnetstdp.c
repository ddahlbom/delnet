#include <stdio.h>
#include <stdlib.h>

#include "delnet.h"


/*************************************************************
 *  Macros
 *************************************************************/
#define SPIKE_BLOCK_SIZE 32768


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
spikerecord *sr_init()
{
	spikerecord *rec;
	rec = malloc(sizeof(spikerecord));
	rec->head = malloc(sizeof(spikeblock));
	rec->head->max_spikes = SPIKE_BLOCK_SIZE;
	rec->head->num_spikes = 0;
	rec->head->spikes = malloc(sizeof(spike)*SPIKE_BLOCK_SIZE);
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
		spikeblock *new = malloc(sizeof(spikeblock));
		new->max_spikes = SPIKE_BLOCK_SIZE;
		new->num_spikes = 0;
		new->spikes = malloc(sizeof(spike)*SPIKE_BLOCK_SIZE);
		new->next = sr->head;
		sr->head = new;
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

	spikes_all = malloc(sizeof(spike)*num_spikes);

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
	spikeblock *new;
	while (curblock != 0) {
		free(curblock->spikes);
		new = curblock->next;
		free(curblock);
		curblock = new;
	}
	free(sr);
}

/*************************************************************
 *  Main
 *************************************************************/

/**
 * @brief Simulation with delnet library
 *
 */
int main()
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

	/* trial parameters */
	fs = 1000.0;
	dur = 1.0;
	p_contact = 0.1;
	n = 1000;
	tau_pre = 0.02;
	tau_post = 0.02;
	a_pre = 0.12;
	a_post = 0.1;
	synbump = 0.000001;
	synmax = 10.0;

	/* derived parameters */
	n_exc = n*0.8;
	n_inh = n*0.2;
	n = n_exc + n_inh;  // in case of rounding issue
	dt = 1.0/fs;
	numsteps = dur/dt;

	/* set up graph */
	g = dn_blobgraph(n, p_contact, 20);
	for (i=n_exc; i<n; i++) 			// only last 200 rows
	for (j=0; j<n; j++) { 				
		g[i*n+j] = g[i*n+j] != 0 ? 1 : 0; 	// 1 ms delay for inh
	}

	/* generate delay network */
	dn = dn_delnetfromgraph(g, n);

	/* initialize neuron and synapse state  */
	neuron *neurons = malloc(sizeof(neuron)*n);
	FLOAT_T *trace_post = calloc(n_exc, sizeof(FLOAT_T));
	FLOAT_T *spike_post = calloc(n_exc, sizeof(FLOAT_T));
	FLOAT_T *trace_pre[n_exc]; 	// pack this
	FLOAT_T *spike_pre[n_exc]; 	// and following
	FLOAT_T *synapses[n]; 		// for speed?

	for (i=0; i<n_exc; i++) {
		neurons[i].v = g_v_default;
		neurons[i].u = g_u_default;
		neurons[i].a = g_a_exc;
		neurons[i].d = g_d_exc;
		trace_pre[i] = calloc(dn->nodes[i].num_in, sizeof(FLOAT_T));
		spike_pre[i] = calloc(dn->nodes[i].num_in, sizeof(FLOAT_T));
		synapses[i]  = malloc(sizeof(FLOAT_T)*dn->nodes[i].num_in);
		for(j=0; j<dn->nodes[i].num_in; j++)
			synapses[i][j] = g_w_exc;
	}

	for (i=n_exc; i<n; i++) {
		neurons[i].v = g_v_default;
		neurons[i].u = g_u_default;
		neurons[i].a = g_a_inh;
		neurons[i].d = g_d_inh;
		synapses[i]  = malloc(sizeof(FLOAT_T)*dn->nodes[i].num_in);
		for(j=0; j<dn->nodes[i].num_in; j++)
			synapses[i][j] = g_w_inh;
	}
	
	/* start simulation */
	FLOAT_T *neuroninputs, inval, outval;
	for (i=0; i<numsteps; i++) {
		t = dt*i;
		if (i%1000 == 0) {
			printf("Time: %f\n", t);
		}
		for (k=0; k<n; k++) {
			/* get inputs to neuron */		
			neuroninputs = dn_getinputaddress(k, dn);
			inval = 0.0;
			for (j=0; j < dn->nodes[k].num_in; j++)
				inval += *(neuroninputs+j);

			/* update synapse traces */
			if (k < n_exc) {
				for (j=0; j < dn->nodes[k].num_in; j++) {
					spike_pre[k][j] = neuroninputs[j];
					trace_pre[k][j] = trace_pre[k][j]*(1.0 - (dt/tau_pre)) +
									  spike_pre[k][j];
				}
			}

			/* random input -- consider placing earlier */
			//if (unirand() < 1.0/n)
			//	inval += 20.0 * (fs/1000.0);
			if (k == 0) 
				inval += 20.0;

			/* update neuron state */
			neurons[k].v += 500.0 * dt * (( 0.04 * neurons[k].v + 5.0) *
							neurons[k].v + 140.0 - neurons[k].u + inval);
			neurons[k].v += 500.0 * dt * (( 0.04 * neurons[k].v + 5.0) *
							neurons[k].v + 140.0 - neurons[k].u + inval);
			neurons[k].u += 1000.0 * dt * neurons[k].a *
								(0.2 * neurons[k].v - neurons[k].u);

			/* check if spiked and calculate output */
			outval = 0.0;
			if (neurons[k].v >= 30.0) {
				sr_save_spike(sr, k, t);
				outval = 1.0;
				neurons[k].v = -65.0;
				neurons[k].u += neurons[k].d;
				numspikes += 1;
			}

			/* update neuron trace */		
			if (k < n_exc) {
				spike_post[k] = outval;
				trace_post[k] = trace_post[k]*(1.0 - (dt/tau_post)) +
								spike_post[k];
			}

			/* update synapse strengths */
			if (k < n_exc) {
				for (j=0; j < dn->nodes[k].num_in; j++) {
					synapses[k][j] = synapses[k][j] + synbump +
							dt * (a_post * trace_pre[k][j] * spike_post[k] -
								  a_pre * trace_post[k] * spike_pre[k][j]);
					synapses[k][j] = synapses[k][j] < 0.0 ? 0.0 : synapses[k][j];
					synapses[k][j] = synapses[k][j] > synmax ? synmax : synapses[k][j];
				}
			}

			/* push the output into the buffer */
			dn_pushoutput(outval, k, dn);
		}

		/* advance the buffer */
		dn_vec_float valspostpush = dn_getinputvec(dn);
		//char* valsstr = dn_vectostr(valspostpush);	
		//printf("%s\n", valsstr);
		free(valspostpush.data);
		//free(valsstr);
		dn_advance(dn);
	}

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
	for (i=0; i<n_exc; i++) {
		free(trace_pre[i]);
		free(spike_pre[i]);
		free(synapses[i]);
	}

	for (i=n_exc; i<n; i++)
		free(synapses[i]);

	return 0;
}
