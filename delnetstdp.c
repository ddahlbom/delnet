#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "delnet.h"
#include "spkrcd.h"


/*************************************************************
 *  Macros
 *************************************************************/


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
	spikerecord *sr = sr_init("delnetstdp.dat", SPIKE_BLOCK_SIZE);
	clock_t t_start, t_finish;
	double gettinginputs, updatingsyntraces, updatingneurons, spikechecking,
			updatingneutraces, updatingsynstrengths, pushingoutput,
			advancingbuffer;

	/* trial parameters */
	fs = 1000.0;
	dur = 1.0;
	p_contact = 0.1;
	n = 1000;
	tau_pre = 0.02;
	tau_post = 0.02;
	a_pre = 0.12;
	a_post = 0.1;
	synbump = 0.00000;
	synmax = 10.0;

	/* process CLI inputs */
	switch (argc) {
		case 4:
			n = atoi(argv[1]);
			p_contact = atof(argv[2]);
			dur = atof(argv[3]);
		case 3:
			n = atoi(argv[1]);
			p_contact = atof(argv[2]);
			break;
		case 2:
			n = atoi(argv[1]);
			break;
	}

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
	printf("tau_post:\t\t%f\n", tau_post);
	printf("A_post:\t\t\t%f\n", a_post);
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
	FLOAT_T *trace_pre; 	// pack this
	FLOAT_T *spike_pre; 	// and following
	FLOAT_T *synapses; 		// for speed?

	IDX_T *offsets = malloc(sizeof(IDX_T)*n);
	IDX_T numsyn_tot=0, numsyn_exc;

	for (i=0; i<n_exc; i++) {
		neurons[i].v = g_v_default;
		neurons[i].u = g_u_default;
		neurons[i].a = g_a_exc;
		neurons[i].d = g_d_exc;
		offsets[i] = numsyn_tot;
		numsyn_tot += dn->nodes[i].num_in;
	}
	numsyn_exc = numsyn_tot;

	for (i=n_exc; i<n; i++) {
		neurons[i].v = g_v_default;
		neurons[i].u = g_u_default;
		neurons[i].a = g_a_inh;
		neurons[i].d = g_d_inh;
		offsets[i] = numsyn_tot;
		numsyn_tot += dn->nodes[i].num_in;
	}

	trace_pre = calloc(numsyn_exc, sizeof(FLOAT_T));		
	spike_pre = calloc(numsyn_exc, sizeof(FLOAT_T));
	synapses = calloc(numsyn_tot, sizeof(FLOAT_T));

	for (i=0; i<numsyn_exc; i++)
		synapses[i] = g_w_exc;
	for (; i<numsyn_tot; i++)
		synapses[i] = g_w_inh;
	
	/* profiling variables */
	gettinginputs 		= 0;
	updatingsyntraces 	= 0;
	updatingneurons 	= 0;
	spikechecking 		= 0;
	updatingneutraces 	= 0;
	updatingsynstrengths = 0;
	pushingoutput 		= 0;
	advancingbuffer 	= 0;

	/* simulation local vars */
	FLOAT_T *neuroninputs, *invals, *outvals;
	invals = malloc(sizeof(FLOAT_T)*n);
	outvals = malloc(sizeof(FLOAT_T)*n);

	/* start simulation */
	for (i=0; i<numsteps; i++) {

		/* calculate time update */
		t = dt*i;
		if (i%1000 == 0) {
			printf("Time: %f\n", t);
		}


		/* get neuron inputs from buffer */
		t_start = clock();
		neuroninputs = dn_getinputaddress(0,dn); //dn->outputs
		for (k=0; k<n; k++) {
			/* get inputs to neuron */		
			invals[k] = 0.0;
			for (j=0; j < dn->nodes[k].num_in; j++) {
				invals[k] += neuroninputs[offsets[k]+j] * synapses[offsets[k]+j];
			}

			/* random input -- consider placing earlier */
			if (unirand() < 1.0/n)
				invals[k] += 20.0 * (fs/1000.0);
		}
		t_finish = clock();
		gettinginputs += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update synapse traces (local variable for STDP) */
		t_start = clock();

		for (k=0; k<n_exc; k++) {
			for (j=0; j < dn->nodes[k].num_in; j++) {
				spike_pre[offsets[k]+j] = neuroninputs[offsets[k]+j];
				trace_pre[offsets[k]+j] = trace_pre[offsets[k]+j]*(1.0 - (dt/tau_pre)) +
								  spike_pre[offsets[k]+j];
			}
		}

		t_finish = clock();
		updatingsyntraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update neuron state */
		for (k=0; k<n; k++) {
			neurons[k].v += 500.0 * dt * (( 0.04 * neurons[k].v + 5.0) *
							neurons[k].v + 140.0 - neurons[k].u + invals[k]);
			neurons[k].v += 500.0 * dt * (( 0.04 * neurons[k].v + 5.0) *
							neurons[k].v + 140.0 - neurons[k].u + invals[k]);
			neurons[k].u += 1000.0 * dt * neurons[k].a *
								(0.2 * neurons[k].v - neurons[k].u);
		}
		t_finish = clock();
		updatingneurons += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


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
		spikechecking += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


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
		updatingneutraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update synapse strengths */
		t_start = clock();
		for (k=0; k<n_exc; k++) {
			for (j=0; j < dn->nodes[k].num_in; j++) {
				synapses[offsets[k]+j] = synapses[offsets[k]+j] + synbump +
						dt * (a_post * trace_pre[offsets[k]+j] * spike_post[k] -
							  a_pre * trace_post[k] * spike_pre[offsets[k]+j]);
				synapses[offsets[k]+j] = synapses[offsets[k]+j] < 0.0 ? 0.0 : synapses[offsets[k]+j];
				synapses[offsets[k]+j] = synapses[offsets[k]+j] > synmax ? synmax : synapses[offsets[k]+j];
			}
		}
		t_finish = clock();
		updatingsynstrengths += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* push the output into the buffer */
		t_start = clock();
		for (k=0; k<n; k++) {
			dn_pushoutput(outvals[k], k, dn);
		}
		t_finish = clock();
		pushingoutput += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* advance the buffer */
		t_start = clock();
		dn_advance(dn);
		t_finish = clock();
		advancingbuffer += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
	}

	/* performance analysis */
	double cycletime, cumtime = 0.0;

	printf("----------------------------------------\n");

	cycletime = 1000.0*gettinginputs/numsteps;
	cumtime += cycletime;
	printf("Getting inputs:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*updatingsyntraces/numsteps;
	cumtime += cycletime;
	printf("Update syntraces:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*updatingneurons/numsteps;
	cumtime += cycletime;
	printf("Update neurons:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*spikechecking/numsteps;
	cumtime += cycletime;
	printf("Check spiked:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*updatingneutraces/numsteps;
	cumtime += cycletime;
	printf("Update neurtrace:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*updatingsynstrengths/numsteps;
	cumtime += cycletime;
	printf("Update syn strength:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*pushingoutput/numsteps;
	cumtime += cycletime;
	printf("Pushing buffer:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*advancingbuffer/numsteps;
	cumtime += cycletime;
	printf("Advancing buffer:\t %f (ms)\n", cycletime);

	printf("Total cycle time:\t %f (ms)\n", cumtime);
	printf("\nTime per second: \t %f (ms)\n", cumtime*fs);

	/* Save spikes */
	//FILE *spike_file;
	//spike_file = fopen( "delnetstdp.dat", "w" );
	//spike *firings = sr_spike_summary(sr);
	//for (i=0; i<numspikes; i++)
	//	fprintf(spike_file, "%f  %d\n", firings[i].time, firings[i].neuron);
	//fclose(spike_file);
	sr_close(sr);


	/* Clean up */
	dn_freedelnet(dn);
	free(g);
	free(trace_post);
	free(spike_post);
	free(neurons);
	free(trace_pre);
	free(spike_pre);
	free(synapses);
	free(invals);
	free(outvals);



	return 0;
}
