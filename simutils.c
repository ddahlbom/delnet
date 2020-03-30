#include <math.h>
#include <stdlib.h>

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
								dn_delaynet *dn, IDX_T *offsets, FLOAT_T dt,
								trialparams *p)
{
	size_t k, j;
	FLOAT_T *neuroninputs;

	for (k=0; k<dn->num_nodes; k++) {
		for (j=0; j < dn->nodes[k].num_in; j++) {
			neuroninputs = dn_getinputaddress(k,dn);
			spike_pre[offsets[k]+j] = neuroninputs[j];
			traces_syn[offsets[k]+j] = traces_syn[offsets[k]+j]*(1.0 - (dt/p->tau_pre)) +
				spike_pre[offsets[k]+j];
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
		if (sourceidx[dn->nodes[k].idx_outbuf+j] < numsyn_exc) {
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
