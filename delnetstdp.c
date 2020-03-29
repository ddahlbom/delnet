#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "delnet.h"
#include "spkrcd.h"
#include "paramutils.h"


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

FLOAT_T g_a_inh  = 0.1;
FLOAT_T g_d_inh  = 2.0;



/*************************************************************
 *  Structs
 *************************************************************/
typedef struct neuron_s {
	FLOAT_T v;
	FLOAT_T u;
	FLOAT_T a;
	FLOAT_T d;
} neuron;

typedef struct trialparams_s {
	double fs;
	double dur;
	double num_neurons;
	double p_contact;
	double p_exc;
	double tau_pre;
	double tau_post;
	double a_pre;
	double a_post;
	double synmax;
	double w_exc;
	double w_inh;
	double lambda;
	double maxdelay;
} trialparams;

/*************************************************************
 *  Functions
 *************************************************************/

/* -------------------- Parameter Setting -------------------- */
void setdefaultparams(trialparams *p)
{
	p->fs = 1000.0;
	p->dur = 1.0;
	p->p_contact = 0.1;
	p->p_exc = 0.8;
	p->lambda = 3.00;
	p->num_neurons = 1000;
	p->tau_pre = 0.02;
	p->tau_post = 0.02;
	p->a_pre = 0.12;
	p->a_post = 0.1;
	p->synmax = 10.0;
	p->w_exc  = 6.0;
	p->w_inh = -5.0;
	p->maxdelay = 20.0;
}

void readparameters(trialparams *p, char *filename)
{
	paramlist *pl = pl_readparams(filename);	

	p->fs = pl_getvalue(pl, "fs");
	p->dur = pl_getvalue(pl, "dur");
	p->p_contact = pl_getvalue(pl, "p_contact");
	p->p_exc = pl_getvalue(pl, "p_exc");
	p->lambda = pl_getvalue(pl, "lambda");
	p->num_neurons = pl_getvalue(pl, "num_neurons");
	p->tau_pre = pl_getvalue(pl, "tau_pre");
	p->tau_post = pl_getvalue(pl, "tau_post");
	p->a_pre = pl_getvalue(pl, "a_pre");
	p->a_post = pl_getvalue(pl, "a_post");
	p->synmax = pl_getvalue(pl, "synmax");
	p->w_exc  = pl_getvalue(pl, "w_exc");
	p->w_inh = pl_getvalue(pl, "w_inh");
	p->maxdelay = pl_getvalue(pl, "maxdelay");
}

void printparameters(trialparams p) {
	/* print trial parameters */
	printf("----------------------------------------\n");
	printf("Sampling Frequency: \t%f\n", p.fs);
	printf("Duration: \t\t%f\n", p.dur);
	printf("Number of nodes: \t%d\n", (int) p.num_neurons);
	printf(" 	Excitatory: \t%d\n", (int) (p.num_neurons*p.p_exc));
	printf(" 	Inhibitory: \t%d\n", (int) (p.num_neurons-p.num_neurons*p.p_exc) );
	printf("Probability of contact:\t%f\n", p.p_contact);
	printf("tau_pre:\t\t%f\n", p.tau_pre);
	printf("A_pre:\t\t\t%f\n", p.a_pre);
	printf("tau_post:\t\t%f\n", p.tau_post);
	printf("A_post:\t\t\t%f\n", p.a_post);
	printf("Max synapse strength:\t%lf\n", p.synmax);
	printf("Exc syn strength:\t%lf\n", p.w_exc);
	printf("Inh syn strength:\t%lf\n", p.w_inh);
	printf("Noise density:\t\t%lf\n", p.lambda);
	printf("Max delay (ms):\t\t%lf\n", p.maxdelay);
	printf("----------------------------------------\n");
}

void analyzeconnectivity(unsigned int *g, unsigned int n,
							unsigned int n_exc, trialparams p) 
{
	size_t i, j;
	double count = 0;
	double cumdur = 0;
	for (i=0; i<n_exc; i++)
	for (j=0; j<n; j++) {
		cumdur += g[i*n+j];
		count += g[i*n+j] != 0 ? 1 : 0.0;
	}
	printf("Average delay line duration (exc): %f (ms)\n", (cumdur/count)*(1000.0/p.fs) );
	for (i=n_exc; i<n; i++)
	for (j=0; j<n; j++) {
		count += g[i*n+j] != 0 ? 1 : 0.0;
	}
	printf("Average connections per neuron: %f\n", count/((double) n));
}

/* -------------------- Initialization Functions -------------------- */
void neuron_set(neuron *n, FLOAT_T v, FLOAT_T u, FLOAT_T a, FLOAT_T d)
{
	n->v = v;
	n->u = u;
	n->a = a;
	n->d = d;
}

/* -------------------- Graph Generation -------------------- */

unsigned int *iblobgraph(trialparams *p)
{

	unsigned int *g, n, n_exc, maxdelay_n;
	size_t i, j;
	double thresh;

	n = p->num_neurons;
	n_exc = (p->num_neurons*p->p_exc);
	thresh = p->p_contact * ((float) n)/((float) n_exc);
	maxdelay_n = (p->maxdelay/1000.0) * p->fs; // since delay in ms


	g = dn_blobgraph(n, p->p_contact, maxdelay_n);
	for (i=n_exc; i<n; i++) { 			// only last 200 rows
		for (j=0; j<n_exc; j++) { 				
			if (unirand() < thresh) 
				g[i*n+j] = 1;
			else
				g[i*n+j] = 0;
		}
		for (j=n_exc; j<n; j++) 
			g[i*n+j] = 0;
	}
	return g;
}


/* -------------------- Random Sampling -------------------- */
double expsampl(double lambda) {
	return -log( (((double) rand()) / ((double) RAND_MAX + 1.0)))/lambda;
}


/*-------------------- Kernels -------------------- */
void sim_getinputs(FLOAT_T *invals, dn_delaynet *dn, FLOAT_T *synapses)
{
	size_t k,j;
	FLOAT_T *neuroninputs;
	for (int k=0; k<dn->num_nodes; k++) {
		// get inputs to neuron (outputs of delaylines)
		invals[k] = 0.0;
		neuroninputs = dn_getinputaddress(k,dn); //dn->outputs
		for (j=0; j < dn->nodes[k].num_in; j++) {
			//invals[k] += neuroninputs[j] * synapses[ offsets[k]+j ];
			invals[k] += neuroninputs[j] * synapses[ dn->nodes[k].idx_oi+j ];
		}
	}
}


/* -------------------- Neuron Equations -------------------- */
static inline FLOAT_T f1(FLOAT_T v, FLOAT_T u, FLOAT_T input) {
	return (0.04*v + 5.0)*v + 140.0 - u + input;
}

static inline FLOAT_T f2(FLOAT_T v, FLOAT_T u, FLOAT_T input, FLOAT_T a) {
	return a*(0.2*v - u);
}

void neuronupdate_rk4(FLOAT_T *v, FLOAT_T *u, FLOAT_T input, FLOAT_T a, FLOAT_T h) {
	FLOAT_T K1, K2, K3, K4, L1, L2, L3, L4, half_h, sixth_h;

	half_h = h*0.5;
	sixth_h = h/6.0;
	
	K1 = f1(*v, *u, input);
	L1 = f2(*v, *u, input, a);
	K2 = f1(*v + half_h*K1, *u + half_h*L1, input); 
	L2 = f2(*v + half_h*K1, *u + half_h*L1, input, a);
	K3 = f1(*v + half_h*K2, *u + half_h*L2, input); 
	L3 = f2(*v + half_h*K2, *u + half_h*L2, input, a);
	K4 = f1(*v + h*K3, *u + h*L3, input);
	L4 = f2(*v + h*K3, *u + h*L3, input, a);

	*v = *v + sixth_h * (K1 + 2*K2 + 2*K3 + K4);
	*u = *u + sixth_h * (L1 + 2*L2 + 2*L3 + L4); 
}


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
	FLOAT_T *trace_post = calloc(n, sizeof(FLOAT_T));
	FLOAT_T *spike_post = calloc(n, sizeof(FLOAT_T));
	FLOAT_T *trace_pre; 	// pack this
	FLOAT_T *spike_pre; 	// and following
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

	trace_pre = calloc(numsyn_tot, sizeof(FLOAT_T));		
	spike_pre = calloc(numsyn_tot, sizeof(FLOAT_T));
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
	FLOAT_T *neuroninputs, *invals, *outvals; 
	IDX_T *sourceidx, *destidx;
	invals = calloc(n, sizeof(FLOAT_T));
	outvals = calloc(n, sizeof(FLOAT_T));

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


	/* -------------------- start simulation -------------------- */
	printf("----------------------------------------\n");
	for (i=0; i<numsteps; i++) {

		/* calculate time update */
		t = dt*i;
		if (i%1000 == 0)
			printf("Time: %f\n", t);


		/* get neuron inputs from buffer */
		t_start = clock();

		/*
		for (k=0; k<n; k++) {
			// get inputs to neuron
			invals[k] = 0.0;
			neuroninputs = dn_getinputaddress(k,dn); //dn->outputs
			for (j=0; j < dn->nodes[k].num_in; j++) {
				invals[k] += neuroninputs[j] * synapses[ offsets[k]+j ];
			}
		}
		*/
		sim_getinputs(invals, dn, synapses);

		//  random input
		for (k=0; k<n; k++) {
			if (nextrand[k] < t) {
				invals[k] += 20.0 * (p.fs/1000); 
				nextrand[k] += expsampl(p.lambda);
				numrandspikes += 1;
			}
		}
		t_finish = clock();
		gettinginputs += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update synapse traces */
		t_start = clock();
		for (k=0; k<n; k++) {
			for (j=0; j < dn->nodes[k].num_in; j++) {
				neuroninputs = dn_getinputaddress(k,dn);
				spike_pre[offsets[k]+j] = neuroninputs[j];
				trace_pre[offsets[k]+j] = trace_pre[offsets[k]+j]*(1.0 - (dt/p.tau_pre)) +
					spike_pre[offsets[k]+j];
			}
		}
		t_finish = clock();
		updatingsyntraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update neuron state */
		t_start = clock();
		for (k=0; k<n; k++)
			neuronupdate_rk4(&neurons[k].v, &neurons[k].u, invals[k], neurons[k].a, 1000.0/p.fs);
		t_finish = clock();
		updatingneurons += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


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


		/* push the output into the buffer */
		t_start = clock();
		for (k=0; k<n; k++)
			dn_pushoutput(outvals[k], k, dn);
		t_finish = clock();
		pushingoutput += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update neuron traces */
		t_start = clock();
		for (k=0; k<n; k++) { 		
			spike_post[k] = outvals[k];
			trace_post[k] = trace_post[k]*(1.0 - (dt/p.tau_post)) + spike_post[k];
		}
		t_finish = clock();
		updatingneutraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;


		/* update synapses */
		t_start = clock();
		for (k=0; k<n; k++) {
			for (j=0; j < dn->nodes[k].num_in; j++) {
				if (sourceidx[offsets[k]+j] < numsyn_exc) {
					synapses[offsets[k]+j] = synapses[offsets[k]+j] +
							dt * (p.a_post * trace_pre[offsets[k]+j] * spike_post[k] -
								  p.a_pre * trace_post[k] * spike_pre[offsets[k]+j]);
					// clamp value	
					synapses[offsets[k]+j] = synapses[offsets[k]+j] < 0.0 ? 
												0.0 : synapses[offsets[k]+j];
					synapses[offsets[k]+j] = synapses[offsets[k]+j] > p.synmax ?
												p.synmax : synapses[offsets[k]+j];
				}
			}
		}
		t_finish = clock();
		updatingsynstrengths += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;

		/* advance the buffer */
		t_start = clock();
		dn_advance(dn);
		t_finish = clock();
		advancingbuffer += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
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
	free(trace_post);
	free(spike_post);
	free(neurons);
	free(trace_pre);
	free(spike_pre);
	free(synapses);
	free(invals);
	free(outvals);
	free(offsets);
	free(nextrand);
	free(sourceidx);
	free(destidx);

	return 0;
}
