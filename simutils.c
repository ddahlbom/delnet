#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "delnet.h"
#include "simutils.h"
#include "paramutils.h"
#include "spkrcd.h"

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
	p->randspikesize=20.0;
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
	p->randspikesize = pl_getvalue(pl, "randspikesize");

	free(pl);
}

void printparameters(trialparams p)
{
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
double expsampl(double lambda)
{
	return -log( (((double) rand()) / ((double) RAND_MAX + 1.0)))/lambda;
}




/* -------------------- Neuron Equations -------------------- */
static inline FLOAT_T f1(FLOAT_T v, FLOAT_T u, FLOAT_T input) {
	return (0.04*v + 5.0)*v + 140.0 - u + input;
}

static inline FLOAT_T f2(FLOAT_T v, FLOAT_T u, FLOAT_T a) {
	return a*(0.2*v - u);
}

void neuronupdate_rk4(FLOAT_T *v, FLOAT_T *u, FLOAT_T input, FLOAT_T a, FLOAT_T h) {
	FLOAT_T K1, K2, K3, K4, L1, L2, L3, L4, half_h, sixth_h;

	half_h = h*0.5;
	sixth_h = h/6.0;
	
	//K1 = f1(*v, *u, input);
	//L1 = f2(*v, *u, input, a);
	//K2 = f1(*v + half_h*K1, *u + half_h*L1, input); 
	//L2 = f2(*v + half_h*K1, *u + half_h*L1, input, a);
	//K3 = f1(*v + half_h*K2, *u + half_h*L2, input); 
	//L3 = f2(*v + half_h*K2, *u + half_h*L2, input, a);
	//K4 = f1(*v + h*K3, *u + h*L3, input);
	//L4 = f2(*v + h*K3, *u + h*L3, input, a);
	K1 = f1(*v, *u, 0.0);
	L1 = f2(*v, *u, a);

	K2 = f1(*v + half_h*K1, *u + half_h*L1, 0.0); 
	L2 = f2(*v + half_h*K1, *u + half_h*L1, a);

	K3 = f1(*v + half_h*K2, *u + half_h*L2, 0.0);
	L3 = f2(*v + half_h*K2, *u + half_h*L2, a);

	K4 = f1(*v + h*K3, *u + h*L3, 0.0);
	L4 = f2(*v + h*K3, *u + h*L3, a);

	//*v = *v + sixth_h * (K1 + 2*K2 + 2*K3 + K4);
	//*u = *u + sixth_h * (L1 + 2*L2 + 2*L3 + L4); 

	*v = *v + sixth_h * (K1 + 2*K2 + 2*K3 + K4) + input;
	*u = *u + sixth_h * (L1 + 2*L2 + 2*L3 + L4); 
}


/*-------------------- Kernels -------------------- */
void sim_getinputs(FLOAT_T *neuroninputs, dn_delaynet *dn, FLOAT_T *synapses)
{
	size_t k,j;
	FLOAT_T *delayoutputs;
	for (k=0; k<dn->num_nodes; k++) {
		// get inputs to neuron (outputs of delaylines)
		neuroninputs[k] = 0.0;
		delayoutputs = dn_getinputaddress(k,dn); //dn->outputs
		for (j=0; j < dn->nodes[k].num_in; j++) {
			neuroninputs[k] += delayoutputs[j] * synapses[ dn->nodes[k].idx_outbuf+j ];
		}
	}
}

unsigned int sim_poisnoise(FLOAT_T *neuroninputs, FLOAT_T *nextrand, FLOAT_T t, trialparams *p)
{
	//  random input
	unsigned int num = 0, k;
	for (k=0; k<p->num_neurons; k++) {
		if (nextrand[k] < t) {
			//neuroninputs[k] += p->randspikesize * (p->fs/1000); 
			neuroninputs[k] += p->randspikesize;
			nextrand[k] += expsampl(p->lambda);
			num += 1;
		}
	}
	return num;
}

void sim_updateneurons(neuron *neurons, FLOAT_T *neuroninputs, trialparams *p)
{
	size_t k;
	for (k=0; k<p->num_neurons; k++) {
		neuronupdate_rk4(&neurons[k].v, &neurons[k].u, neuroninputs[k],
							neurons[k].a, 1000.0/p->fs);
	}
}

unsigned int sim_checkspiking(neuron *neurons, FLOAT_T *neuronoutputs,
								unsigned int n, FLOAT_T t, spikerecord *sr)
{
	size_t k;
	unsigned int numspikes=0;
	for (k=0; k<n; k++) {
		neuronoutputs[k] = 0.0;
		if (neurons[k].v >= 30.0) {
			sr_save_spike(sr, k, t);
			neuronoutputs[k] = 1.0;
			neurons[k].v = -65.0;
			neurons[k].u += neurons[k].d;
			numspikes += 1;
		}
	}
	return numspikes;
}

void sim_updatesynapsetraces(FLOAT_T *traces_syn, FLOAT_T *spike_pre,
								dn_delaynet *dn, FLOAT_T dt,
								trialparams *p)
{
	size_t k, j;
	FLOAT_T *neuroninputs;

	for (k=0; k<dn->num_nodes; k++) {
		for (j=0; j < dn->nodes[k].num_in; j++) {
			neuroninputs = dn_getinputaddress(k,dn);
			spike_pre[dn->nodes[k].idx_outbuf +j] = neuroninputs[j];
			traces_syn[dn->nodes[k].idx_outbuf +j] = traces_syn[dn->nodes[k].idx_outbuf +j]*(1.0 - (dt/p->tau_pre)) +
				spike_pre[dn->nodes[k].idx_outbuf +j];
		}
	}
}


void sim_updateneurontraces(FLOAT_T *traces_neu, FLOAT_T *neuronoutputs, IDX_T n,
								FLOAT_T dt, trialparams *p) 
{
	size_t k;
	for (k=0; k<n; k++) { 		
		traces_neu[k] = traces_neu[k]*(1.0 - (dt/p->tau_post)) + neuronoutputs[k];
	}
}

void sim_updatesynapses(FLOAT_T *synapses, FLOAT_T *traces_syn, FLOAT_T *traces_neu, 
							FLOAT_T *neuronoutputs, dn_delaynet *dn, IDX_T *sourceidx,
							FLOAT_T dt, unsigned int numsyn_exc, trialparams *p)
{
	size_t k, j;
	FLOAT_T *synapseoutputs = dn->outputs;
	for (k=0; k<p->num_neurons; k++) 
	for (j=0; j < dn->nodes[k].num_in; j++) {
		// only update excitatory synapses
		//if (sourceidx[dn->nodes[k].idx_outbuf+j] < numsyn_exc) {
		if (synapses[dn->nodes[k].idx_outbuf+j] > 0) {
			synapses[dn->nodes[k].idx_outbuf+j] = synapses[dn->nodes[k].idx_outbuf+j] +
					dt * (p->a_post * traces_syn[dn->nodes[k].idx_outbuf+j] * neuronoutputs[k] -
						  p->a_pre * traces_neu[k] * synapseoutputs[dn->nodes[k].idx_outbuf+j]);
			// clamp value	
			synapses[dn->nodes[k].idx_outbuf+j] = synapses[dn->nodes[k].idx_outbuf+j] < 0.0 ? 
										0.0 : synapses[dn->nodes[k].idx_outbuf+j];
			synapses[dn->nodes[k].idx_outbuf+j] = synapses[dn->nodes[k].idx_outbuf+j] > p->synmax ?
										p->synmax : synapses[dn->nodes[k].idx_outbuf+j];
		}
	}
	
}




void runstdpmodel(sim_model *m, FLOAT_T *input, size_t inputlen,
					FLOAT_T dur, spikerecord *sr, bool profiling) {

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

	/* derived params -- trim later, maybe cruft */
	IDX_T n = m->p.num_neurons;
	FLOAT_T dt = 1.0/m->p.fs;
	IDX_T numsteps = m->p.dur/dt;
	clock_t t_start=clock(), t_finish;

	/* local state for simulation */
	FLOAT_T *neuroninputs, *neuronoutputs; 
	FLOAT_T *nextrand = malloc(sizeof(FLOAT_T)*n);
	neuroninputs = calloc(n, sizeof(FLOAT_T));
	neuronoutputs = calloc(n, sizeof(FLOAT_T));
	unsigned long int numspikes = 0, numrandspikes = 0;
	FLOAT_T offdur = 1.0; // <--- get rid of this after testing!

	for(size_t i=0; i<n; i++)
		nextrand[i] = expsampl(m->p.lambda);

	for (size_t i=0; i<numsteps; i++) {

		/* ---------- calculate time update ---------- */
		FLOAT_T t = dt*i;
		if (i%1000 == 0)
			printf("Time: %f\n", t);


		/* ---------- get delay outputs (neuron inputs) from buffer ---------- */
		if (profiling) t_start = clock();

		sim_getinputs(neuroninputs, m->dn, m->synapses);
		numrandspikes += sim_poisnoise(neuroninputs, nextrand, t, &m->p);

		if (profiling) {
			t_finish = clock();
			gettinginputs += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}

		/* put in forced input */
		if (t < m->p.dur - offdur) {
			for (size_t k=0; k < m->numinputneurons; k++)
				neuroninputs[k] += input[ i % inputlen ];
		}

		/* ---------- update neuron state ---------- */
		if (profiling) t_start = clock();

		sim_updateneurons(m->neurons, neuroninputs, &m->p);

		if (profiling) {
			t_finish = clock();
			updatingneurons += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}

		/* ---------- calculate neuron outputs ---------- */
		if (profiling) t_start = clock();

		numspikes += sim_checkspiking(m->neurons, neuronoutputs, n, t, sr);

		if (profiling) {
			t_finish = clock();
			spikechecking += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- push the neuron output into the buffer ---------- */
		if (profiling) t_start = clock();

		for (size_t k=0; k<n; k++)
			dn_pushoutput(neuronoutputs[k], k, m->dn);

		if (profiling) {
			t_finish = clock();
			pushingoutput += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- update synapse traces ---------- */
		if (profiling) t_start = clock();

		//sim_updatesynapsetraces(traces_syn, spike_pre, dn, offsets, dt, &p);
		sim_updatesynapsetraces(m->traces_syn, m->dn->outputs, m->dn, dt, &m->p);

		if (profiling) {
			t_finish = clock();
			updatingsyntraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}



		/* ---------- update neuron traces ---------- */
		if (profiling) t_start = clock();

		sim_updateneurontraces(m->traces_neu, neuronoutputs, n, dt, &m->p);

		if (profiling) {
			t_finish = clock();
			updatingneutraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- update synapses ---------- */
		if (profiling) t_start = clock();

		sim_updatesynapses(m->synapses, m->traces_syn, m->traces_neu, neuronoutputs,
							m->dn, m->dn->sourceidx, dt, m->numsyn_exc, &m->p);

		if (profiling) {
			t_finish = clock();
			updatingsynstrengths += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* advance the buffer */
		if (profiling) t_start = clock();

		dn_advance(m->dn);

		if (profiling) {
			t_finish = clock();
			advancingbuffer += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}
	}


	/* -------------------- Performance Analysis -------------------- */
	if (profiling) {
		printf("----------------------------------------\n");
		printf("Random input rate: %g\n", ((double) numrandspikes) / (((double) n)*m->p.dur) );
		printf("Firing rate: %g\n", ((double) numspikes) / (((double) n)*m->p.dur) );
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
		printf("\nTime per second: \t %f (ms)\n", cumtime*m->p.fs);
	}

	free(neuroninputs);
	free(neuronoutputs);
	free(nextrand);
}

void sim_savemodel(sim_model *m, char *filename) {
	FILE *f = fopen(filename, "wb");
	
	dn_savedelnet(m->dn, f);

	fwrite(&m->numinputneurons, sizeof(IDX_T), 1, f);
	fwrite(&m->numsyn_exc, sizeof(IDX_T), 1, f);
	fwrite(&m->p, sizeof(trialparams), 1, f);
	fwrite(m->neurons, sizeof(neuron), m->dn->num_nodes, f);
	fwrite(m->traces_neu, sizeof(FLOAT_T), m->dn->num_nodes, f);
	fwrite(m->traces_syn, sizeof(FLOAT_T), m->dn->num_delays, f);
	fwrite(m->synapses, sizeof(FLOAT_T), m->dn->num_delays, f);


	fclose(f);
}

sim_model *sim_loadmodel(char *filename) {
	sim_model *m = malloc(sizeof(sim_model));
	size_t loadsize;
	FILE *f = fopen(filename, "rb");
	
	m->dn = dn_loaddelnet(f);

	loadsize = fread(&m->numinputneurons, sizeof(IDX_T), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->numsyn_exc, sizeof(IDX_T), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->p, sizeof(trialparams), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	m->neurons = malloc(sizeof(neuron)*m->dn->num_nodes);
	loadsize = fread(m->neurons, sizeof(neuron), m->dn->num_nodes, f);
	if (loadsize != m->dn->num_nodes) { printf("Failed to load model.\n"); exit(-1); }

	m->traces_neu = malloc(sizeof(FLOAT_T)*m->dn->num_nodes);
	loadsize = fread(m->traces_neu, sizeof(FLOAT_T), m->dn->num_nodes, f);
	if (loadsize != m->dn->num_nodes) { printf("Failed to load model.\n"); exit(-1); }

	m->traces_syn = malloc(sizeof(FLOAT_T)*m->dn->num_delays);
	loadsize = fread(m->traces_syn, sizeof(FLOAT_T), m->dn->num_delays, f);
	if (loadsize != m->dn->num_delays) { printf("Failed to load model.\n"); exit(-1); }

	m->synapses = malloc(sizeof(FLOAT_T)*m->dn->num_delays);
	loadsize = fread(m->synapses, sizeof(FLOAT_T), m->dn->num_delays, f);
	if (loadsize != m->dn->num_delays) { printf("Failed to load model.\n"); exit(-1); }

	fclose(f);

	return m;
}

void sim_freemodel(sim_model *m) {
	dn_freedelnet(m->dn);
	free(m->neurons);
	free(m->traces_neu);
	free(m->traces_syn);
	free(m->synapses);
	free(m);
}

