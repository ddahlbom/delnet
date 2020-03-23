#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

static inline FLOAT_T f1(FLOAT_T v, FLOAT_T u, FLOAT_T input) {
	return (0.04*v + 5.0)*v + 140.0 - u + input;
}

static inline FLOAT_T f2(FLOAT_T v, FLOAT_T u, FLOAT_T input, FLOAT_T a) {
	return a*(0.2*v - u);
}

void update_rk4(FLOAT_T *v, FLOAT_T *u, FLOAT_T input, FLOAT_T a, FLOAT_T h) {
	FLOAT_T K1, K2, K3, K4, L1, L2, L3, L4, half_h, sixth_h;
	half_h = h*0.5;
	sixth_h = h/6.0;
	K1 = f1(*v, *u, input);
	L1 = f2(*v, *u, input, a);

	K2 = f1(*v+half_h*K1, *u+half_h*L1, input); 
	L2 = f2(*v+half_h*K1, *u+half_h*L1, input, a);

	K3 = f1(*v+half_h*K2, *u+half_h*L2, input); 
	L3 = f2(*v+half_h*K2, *u+half_h*L2, input, a);

	K4 = f1(*v+h*K3, *u+h*L3, input);
	L4 = f2(*v+h*K3, *u+h*L3, input, a);

	*v = *v + sixth_h * (K1 + 2*K2 + 2*K3 + K4);
	*u = *u + sixth_h * (L1 + 2*L2 + 2*L3 + L4); 
}

double expsampl(double lambda) {
	return -log( (((double) rand()) / ((double) RAND_MAX + 1.0)))/lambda;
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
	unsigned long int numspikes = 0, numrandspikes = 0;
	double p_contact, lambda;
	dn_delaynet *dn;
	spikerecord *sr = sr_init("delnetstdp.dat", SPIKE_BLOCK_SIZE);
	clock_t t_start, t_finish;
	double gettinginputs, updatingsyntraces, updatingneurons, spikechecking,
			updatingneutraces, updatingsynstrengths, pushingoutput,
			advancingbuffer;

	srand(1);

	/* trial parameters */
	fs = 1000.0;
	dur = 1.0;
	p_contact = 0.1;
	lambda = 2.0; 	// subject neurons to poissonian noise at lambda Hz
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
	for (i=n_exc; i<n; i++) { 			// only last 200 rows
		for (j=0; j<n_exc; j++) { 				
			if (unirand() < p_contact * ((float) n)/((float) n_exc)) 
				g[i*n+j] = 1;
			else
				g[i*n+j] = 0;
		}
		for (j=n_exc; j<n; j++) 
			g[i*n+j] = 0;
	}

	/* analyze connectivity */
	double count = 0;
	for (i=0; i<n; i++)
	for (j=0; j<n; j++)
		count += g[i*n+j] != 0 ? 1 : 0.0;
	printf("Average connections per neuron: %f\n", count/((double) n));

	/* generate delay network */
	dn = dn_delnetfromgraph(g, n);

	/* initialize neuron and synapse state  */
	neuron *neurons = malloc(sizeof(neuron)*n);
	double *nextrand = malloc(sizeof(double)*n);
	//FLOAT_T *trace_post = calloc(n_exc, sizeof(FLOAT_T));
	//FLOAT_T *spike_post = calloc(n_exc, sizeof(FLOAT_T));
	//FLOAT_T *trace_pre; 	// pack this
	//FLOAT_T *spike_pre; 	// and following
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

	//trace_pre = calloc(numsyn_exc, sizeof(FLOAT_T));		
	//spike_pre = calloc(numsyn_exc, sizeof(FLOAT_T));
	synapses = calloc(numsyn_tot, sizeof(FLOAT_T));

	
	/* profiling variables */
	gettinginputs 		= 0;
	//updatingsyntraces 	= 0;
	updatingneurons 	= 0;
	spikechecking 		= 0;
	//updatingneutraces 	= 0;
	//updatingsynstrengths = 0;
	pushingoutput 		= 0;
	advancingbuffer 	= 0;

	/* simulation local vars */
	FLOAT_T *neuroninputs, *invals, *outvals; 
	IDX_T *invinvidx;
	invals = calloc(n, sizeof(FLOAT_T));
	outvals = calloc(n, sizeof(FLOAT_T));
	invinvidx = calloc(numsyn_tot, sizeof(IDX_T));
	//size_t randnode;
	for (i=0; i<numsyn_tot; i++)
		invinvidx[dn->inverseidx[i]] = i;

	//for (i=0; i<numsyn_exc; i++)
	//	synapses[invinvidx[i]] = g_w_exc;
	//for (; i<numsyn_tot; i++)
	//	synapses[invinvidx[i]] = g_w_inh;
	for (i=0; i<numsyn_exc; i++)
		synapses[dn->inverseidx[i]] = g_w_exc;
	for (; i<numsyn_tot; i++)
		synapses[dn->inverseidx[i]] = g_w_inh;

	for(i=0; i<n; i++)
		nextrand[i] = expsampl(lambda);

	/* start simulation */
	for (i=0; i<numsteps; i++) {

		/* calculate time update */
		t = dt*i;
		if (i%1000 == 0) {
			printf("Time: %f\n", t);
		}


		/* get neuron inputs from buffer */
		t_start = clock();
		//neuroninputs = dn_getinputaddress(0,dn); //dn->outputs
		//randnode = getrandom(n);
		for (k=0; k<n; k++) {
			/* get inputs to neuron */		
			invals[k] = 0.0;
			neuroninputs = dn_getinputaddress(k,dn); //dn->outputs
			for (j=0; j < dn->nodes[k].num_in; j++) {
				invals[k] += neuroninputs[j] * synapses[ offsets[k]+j ];
			}

			/* random input */ 
			//if (unirand() < 1.0/n)
			//	invals[k] += 20.0 * (fs/1000.0);
			if (nextrand[k] < t) {
				invals[k] += 20.0;
				nextrand[k] += expsampl(lambda);
				numrandspikes += 1;
			}
		}
		t_finish = clock();
		gettinginputs += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* check if fired and calculate output */
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


		/* update neuron state */
		//FLOAT_T v_coef = 500.0/fs;
		//FLOAT_T u_coef = 1000.0/fs;
		for (k=0; k<n; k++) {
			//neurons[k].v += v_coef*((0.04*neurons[k].v+5.0)*neurons[k].v + 
			//							140.0 - neurons[k].u + invals[k]);
			//neurons[k].v += v_coef*((0.04*neurons[k].v+5.0)*neurons[k].v +
			//							140.0 - neurons[k].u + invals[k]);
			//neurons[k].u += u_coef*neurons[k].a*(0.2*neurons[k].v-neurons[k].u);
			// try out different numerical scheme, e.g. Runge-Kutta
			update_rk4(&neurons[k].v, &neurons[k].u, invals[k], neurons[k].a, 1000.0/fs);
		}
		t_finish = clock();
		updatingneurons += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


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
	printf("----------------------------------------\n");
	printf("Random input rate: %g\n", ((double) numrandspikes) / (((double) n)*dur) );
	printf("Firing rate: %g\n", ((double) numspikes) / (((double) n)*dur) );
	double cycletime, cumtime = 0.0;

	printf("----------------------------------------\n");

	cycletime = 1000.0*gettinginputs/numsteps;
	cumtime += cycletime;
	printf("Getting inputs:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*updatingneurons/numsteps;
	cumtime += cycletime;
	printf("Update neurons:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*spikechecking/numsteps;
	cumtime += cycletime;
	printf("Check spiked:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*pushingoutput/numsteps;
	cumtime += cycletime;
	printf("Pushing buffer:\t\t %f (ms)\n", cycletime);

	cycletime = 1000.0*advancingbuffer/numsteps;
	cumtime += cycletime;
	printf("Advancing buffer:\t %f (ms)\n", cycletime);

	printf("Total cycle time:\t %f (ms)\n", cumtime);
	printf("\nTime per second: \t %f (ms)\n", cumtime*fs);

	/* Save spikes */
	sr_close(sr);

	/* Clean up delay network */
	dn_freedelnet(dn);

	/* Clean up */
	free(g);
	// free(trace_post);
	// free(spike_post);
	free(neurons);
	// free(trace_pre);
	// free(spike_pre);
	free(synapses);
	free(invals);
	free(outvals);
	free(offsets);
	free(nextrand);

	return 0;
}
