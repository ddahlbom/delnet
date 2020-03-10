#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "delnet.h"


/*************************************************************
 *  Macros
 *************************************************************/
//#define SPIKE_BLOCK_SIZE 32768
#define SPIKE_BLOCK_SIZE 8192


/*************************************************************
 *  Globals
 *************************************************************/
FLOAT_T g_v_default = -65.0;
FLOAT_T g_u_default = -13.0;

FLOAT_T g_a_exc  = 0.02;
FLOAT_T g_d_exc  = 8.0;
FLOAT_T g_w_exc  = 6.0*0.9;

FLOAT_T g_a_inh  = 0.1;
FLOAT_T g_d_inh  = 2.0;
FLOAT_T g_w_inh = -5.0*0.8;


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
double dd_sum_double(double *vals, size_t n) {
	double sum = 0.0;
	for (int k=0; k<n; k++) 
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
	clock_t t_start, t_finish;
	double *gettinginputs, *updatingsyntraces, *updatingneurons, *spikechecking,
			*updatingneutraces, *updatingsynstrengths, *pushingoutput,
			*advancingbuffer;

	/* trial parameters */
	fs = 1000.0;
	dur = 1.0;
	p_contact = 0.09;
	n = 1000;
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
	
	gettinginputs = malloc(sizeof(double)*numsteps*n);
	updatingsyntraces = malloc(sizeof(double)*numsteps*n);
	updatingneurons = malloc(sizeof(double)*numsteps*n);
	spikechecking = malloc(sizeof(double)*numsteps*n);
	updatingneutraces = malloc(sizeof(double)*numsteps*n);
	updatingsynstrengths = malloc(sizeof(double)*numsteps*n);
	pushingoutput = malloc(sizeof(double)*numsteps*n);
	advancingbuffer = malloc(sizeof(double)*numsteps);

	/* start simulation */
	FLOAT_T *neuroninputs, inval, outval;
	for (i=0; i<numsteps; i++) {
		t = dt*i;
		if (i%1000 == 0) {
			printf("Time: %f\n", t);
		}
		for (k=0; k<n; k++) {
			/* get inputs to neuron */		
			t_start = clock();
			neuroninputs = dn_getinputaddress(k, dn);
			inval = 0.0;
			for (j=0; j < dn->nodes[k].num_in; j++) {
				inval += *(neuroninputs+j) * synapses[k][j];
			}
			t_finish = clock();
			gettinginputs[i*n+k] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

			/* update synapse traces */
			t_start = clock();
			if (k < n_exc) {
				for (j=0; j < dn->nodes[k].num_in; j++) {
					spike_pre[k][j] = neuroninputs[j];
					trace_pre[k][j] = trace_pre[k][j]*(1.0 - (dt/tau_pre)) +
									  spike_pre[k][j];
				}
			}
			t_finish = clock();
			updatingsyntraces[i*n+k] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

			/* random input -- consider placing earlier */
			if (unirand() < 1.0/n)
				inval += 20.0 * (fs/1000.0);

			/* update neuron state */
			t_start = clock();
			neurons[k].v += 500.0 * dt * (( 0.04 * neurons[k].v + 5.0) *
							neurons[k].v + 140.0 - neurons[k].u + inval);
			neurons[k].v += 500.0 * dt * (( 0.04 * neurons[k].v + 5.0) *
							neurons[k].v + 140.0 - neurons[k].u + inval);
			neurons[k].u += 1000.0 * dt * neurons[k].a *
								(0.2 * neurons[k].v - neurons[k].u);
			t_finish = clock();
			updatingneurons[i*n+k] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

			/* check if spiked and calculate output */
			t_start = clock();
			outval = 0.0;
			if (neurons[k].v >= 30.0) {
				sr_save_spike(sr, k, t);
				outval = 1.0;
				neurons[k].v = -65.0;
				neurons[k].u += neurons[k].d;
				numspikes += 1;
			}
			t_finish = clock();
			spikechecking[i*n+k] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

			/* update neuron trace */		
			t_start = clock();
			if (k < n_exc) {
				spike_post[k] = outval;
				trace_post[k] = trace_post[k]*(1.0 - (dt/tau_post)) +
								spike_post[k];
			}
			t_finish = clock();
			updatingneutraces[i*n+k] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

			/* update synapse strengths */
			t_start = clock();
			if (k < n_exc) {
				for (j=0; j < dn->nodes[k].num_in; j++) {
					synapses[k][j] = synapses[k][j] + synbump +
							dt * (a_post * trace_pre[k][j] * spike_post[k] -
								  a_pre * trace_post[k] * spike_pre[k][j]);
					synapses[k][j] = synapses[k][j] < 0.0 ? 0.0 : synapses[k][j];
					synapses[k][j] = synapses[k][j] > synmax ? synmax : synapses[k][j];
					//spike_post[k] = 0.0;
				}
			}
			t_finish = clock();
			updatingsynstrengths[i*n+k] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

			/* push the output into the buffer */
			t_start = clock();
			dn_pushoutput(outval, k, dn);
			t_finish = clock();
			pushingoutput[i*n+k] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}

		/* advance the buffer */
		t_start = clock();
		dn_advance(dn);
		t_finish = clock();
		pushingoutput[i*n+k] = ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
	}

	/* performance analysis */
	double cycletime, cumtime = 0.0;

	printf("----------------------------------------\n");

	cycletime = 1000.0*dd_sum_double(gettinginputs, n*numsteps)/numsteps;
	cumtime += cycletime;
	printf("Getting inputs:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(updatingsyntraces, n*numsteps)/numsteps;
	cumtime += cycletime;
	printf("Update syntraces:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(updatingneurons, n*numsteps)/numsteps;
	cumtime += cycletime;
	printf("Update neurons:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(spikechecking, n*numsteps)/numsteps;
	cumtime += cycletime;
	printf("Check spiked:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(updatingneutraces, n*numsteps)/numsteps;
	cumtime += cycletime;
	printf("Update neurtrace:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(updatingsynstrengths, n*numsteps)/numsteps;
	cumtime += cycletime;
	printf("Update syn strength:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*dd_sum_double(pushingoutput, n*numsteps)/numsteps;
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
	for (i=0; i<n_exc; i++) {
		free(trace_pre[i]);
		free(spike_pre[i]);
		free(synapses[i]);
	}

	for (i=n_exc; i<n; i++)
		free(synapses[i]);

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
