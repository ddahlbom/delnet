#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "delnet.h"
#include "spkrcd.h"
#include "paramutils.h"
#include "simutils.h"


/*************************************************************
 *  Macros
 *************************************************************/
#define PROFILING 1

/*************************************************************
 *  Globals
 *************************************************************/
FLOAT_T g_v_default = -65.0;
FLOAT_T g_u_default = -13.0;

FLOAT_T g_a_exc  = 0.02;
FLOAT_T g_d_exc  = 8.0;

FLOAT_T g_a_inh  = 0.1;
FLOAT_T g_d_inh  = 2.0;

/*************************************************************
 *  Main
 *************************************************************/

int main(int argc, char *argv[])
{
	trialparams p;
	FLOAT_T dt, t;
	unsigned int i, j, k, numsteps;
	unsigned int n, n_exc;
	unsigned int *g;
	unsigned long int numspikes = 0, numrandspikes = 0;
	dn_delaynet *dn;
	spikerecord *sr = sr_init("delnetstdp.dat", SPIKE_BLOCK_SIZE);
	clock_t t_start, t_finish;

	srand(1);

	/* set parameters */
	if (argc < 2) {
		printf("No parameter file given.  Using defaults. \n");
		setdefaultparams(&p);
	} else
		readparameters(&p, argv[1]);

	/* derived parameters */
	n = p.num_neurons;
	n_exc = (unsigned int) ( (double) n * p.p_exc);
	dt = 1.0/p.fs;
	numsteps = p.dur/dt;

	/* print parameters */
	printparameters(p);

	/* set up graph */
	g = iblobgraph(&p);

	/* analyze connectivity -- sanity check*/
	analyzeconnectivity(g, n, n_exc, p);

	/* generate delay network */
	dn = dn_delnetfromgraph(g, n);

	/* initialize neuron and synapse state  */
	neuron *neurons = malloc(sizeof(neuron)*n);
	double *nextrand = malloc(sizeof(double)*n);
	FLOAT_T *traces_neu = calloc(n, sizeof(FLOAT_T));
	FLOAT_T *traces_syn; 	// pack this
	FLOAT_T *synapses; 		// for speed?

	IDX_T *offsets = malloc(sizeof(IDX_T)*n);
	IDX_T numsyn_tot=0, numsyn_exc;

	for (i=0; i<n_exc; i++) {
		neuron_set(&neurons[i], g_v_default, g_u_default, g_a_exc, g_d_exc);
		offsets[i] = numsyn_tot;
		numsyn_tot += dn->nodes[i].num_in;
	}
	numsyn_exc = numsyn_tot;
	for (i=n_exc; i<n; i++) {
		neuron_set(&neurons[i], g_v_default, g_u_default, g_a_inh, g_d_inh);
		offsets[i] = numsyn_tot;
		numsyn_tot += dn->nodes[i].num_in;
	}

	traces_syn = calloc(numsyn_tot, sizeof(FLOAT_T));		
	synapses = calloc(numsyn_tot, sizeof(FLOAT_T));
	

	/* profiling variables */
	double gettinginputs, updatingsyntraces, updatingneurons, spikechecking,
			updatingneutraces, updatingsynstrengths, pushingoutput,
			advancingbuffer;

	gettinginputs 		= 0;
	updatingsyntraces 	= 0;
	updatingneurons 	= 0;
	spikechecking 		= 0;
	updatingneutraces 	= 0;
	updatingsynstrengths = 0;
	pushingoutput 		= 0;
	advancingbuffer 	= 0;

	/* simulation local vars */
	FLOAT_T *neuroninputs, *neuronoutputs; 
	IDX_T *sourceidx, *destidx;
	neuroninputs = calloc(n, sizeof(FLOAT_T));
	neuronoutputs = calloc(n, sizeof(FLOAT_T));

	/* index arithmetic */
	sourceidx = calloc(numsyn_tot, sizeof(IDX_T));
	destidx = calloc(numsyn_tot, sizeof(IDX_T));
	for (i=0; i<numsyn_tot; i++) {
		destidx[i] = dn->inverseidx[i];
		sourceidx[dn->inverseidx[i]] = i; 
	}

	/* initialize synapse weights */
	for (i=0; i<numsyn_exc; i++)
		synapses[destidx[i]] = p.w_exc;
	for (; i<numsyn_tot; i++)
		synapses[destidx[i]] = p.w_inh;

	/* initialize random input times */
	for(i=0; i<n; i++)
		nextrand[i] = expsampl(p.lambda);

	printf("----------------------------------------\n");


	/* -------------------- start simulation -------------------- */
	for (i=0; i<numsteps; i++) {

		/* ---------- calculate time update ---------- */
		t = dt*i;
		if (i%1000 == 0)
			printf("Time: %f\n", t);


		/* ---------- get delay outputs (neuron inputs) from buffer ---------- */
		if (PROFILING) t_start = clock();

		sim_getinputs(neuroninputs, dn, synapses);
		numrandspikes += sim_poisnoise(neuroninputs, nextrand, t, &p);

		if (PROFILING) {
			t_finish = clock();
			gettinginputs += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- update neuron state ---------- */
		if (PROFILING) t_start = clock();

		sim_updateneurons(neurons, neuroninputs, &p);

		if (PROFILING) {
			t_finish = clock();
			updatingneurons += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- calculate neuron outputs ---------- */
		if (PROFILING) t_start = clock();

		numspikes += sim_checkspiking(neurons, neuronoutputs, n, t, sr);

		if (PROFILING) {
			t_finish = clock();
			spikechecking += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- push the neuron output into the buffer ---------- */
		if (PROFILING) t_start = clock();

		for (k=0; k<n; k++)
			dn_pushoutput(neuronoutputs[k], k, dn);

		if (PROFILING) {
			t_finish = clock();
			pushingoutput += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- update synapse traces ---------- */
		if (PROFILING) t_start = clock();

		//sim_updatesynapsetraces(traces_syn, spike_pre, dn, offsets, dt, &p);
		sim_updatesynapsetraces(traces_syn, dn->outputs, dn, offsets, dt, &p);

		if (PROFILING) {
			t_finish = clock();
			updatingsyntraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}



		/* ---------- update neuron traces ---------- */
		if (PROFILING) t_start = clock();

		sim_updateneurontraces(traces_neu, neuronoutputs, n, dt, &p);

		if (PROFILING) {
			t_finish = clock();
			updatingneutraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- update synapses ---------- */
		if (PROFILING) t_start = clock();

		sim_updatesynapses(synapses, traces_syn, traces_neu, neuronoutputs,
							dn, sourceidx, dt, numsyn_exc, &p);

		if (PROFILING) {
			t_finish = clock();
			updatingsynstrengths += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* advance the buffer */
		if (PROFILING) t_start = clock();

		dn_advance(dn);

		if (PROFILING) {
			t_finish = clock();
			advancingbuffer += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}
	}


	/* -------------------- Performance Analysis -------------------- */
	printf("----------------------------------------\n");
	printf("Random input rate: %g\n", ((double) numrandspikes) / (((double) n)*p.dur) );
	printf("Firing rate: %g\n", ((double) numspikes) / (((double) n)*p.dur) );
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

	cycletime = 1000.0*pushingoutput/numsteps;
	cumtime += cycletime;
	printf("Pushing buffer:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*updatingsynstrengths/numsteps;
	cumtime += cycletime;
	printf("Updating synapses:\t %f (ms)\n", cycletime);

	cycletime = 1000.0*advancingbuffer/numsteps;
	cumtime += cycletime;
	printf("Advancing buffer:\t %f (ms)\n", cycletime);

	printf("Total cycle time:\t %f (ms)\n", cumtime);
	printf("\nTime per second: \t %f (ms)\n", cumtime*p.fs);
	
	printf("Don't optimize out, please! (%d)\n", destidx[0]);

	/* save synapse weights */
	FILE *f;
	f = fopen("synapses.dat", "w");
	for (k=0; k<numsyn_tot; k++) {
		fprintf(f, "%g\n", synapses[destidx[k]]);
	}
	fclose(f);

	/* Save spikes */
	sr_close(sr);

	/* Clean up delay network */
	dn_freedelnet(dn);

	/* Clean up */
	free(g);
	free(traces_neu);
	free(neurons);
	free(traces_syn);
	free(synapses);
	free(neuroninputs);
	free(neuronoutputs);
	free(offsets);
	free(nextrand);
	free(sourceidx);
	free(destidx);

	return 0;
}
