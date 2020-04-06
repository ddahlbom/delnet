#ifndef SIMUTILS_H
#define SIMUTILS_H

#include "spkrcd.h"
#include <stdbool.h>

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
	double randspikesize;
} trialparams;

typedef struct sim_model_s {
	trialparams p;
	dn_delaynet *dn;
	neuron  *neurons;
	FLOAT_T *traces_neu;
	FLOAT_T *traces_syn;
	FLOAT_T *synapses;
	IDX_T numinputneurons;
	IDX_T numsyn_exc;
} sim_model;

/*************************************************************
 *  Function Declarations
 *************************************************************/

/* parameter bookkeeping */
void setdefaultparams(trialparams *p);
void readparameters(trialparams *p, char *filename);
void printparameters(trialparams p);
void analyzeconnectivity(unsigned int *g, unsigned int n,
							unsigned int n_exc, trialparams p);

/* graph generation*/
unsigned int *iblobgraph(trialparams *p);

/* neuron info */
void neuron_set(neuron *n, FLOAT_T v, FLOAT_T u, FLOAT_T a, FLOAT_T d);
void neuronupdate_rk4(FLOAT_T *v, FLOAT_T *u, FLOAT_T input, FLOAT_T a, FLOAT_T h);

/* other utility functions */
double expsampl(double lambda);

/* kernels */
void sim_getinputs(FLOAT_T *neuroninputs, dn_delaynet *dn, FLOAT_T *synapses);
unsigned int sim_poisnoise(FLOAT_T *neuroninputs, FLOAT_T *nextrand, FLOAT_T t, trialparams *p);
void sim_updateneurons(neuron *neurons, FLOAT_T *neuroninputs, trialparams *p);
unsigned int sim_checkspiking(neuron *neurons, FLOAT_T *neuronoutputs,
								unsigned int n, FLOAT_T t, spikerecord *sr);
void sim_updatesynapsetraces(FLOAT_T *traces_syn, FLOAT_T *spike_pre,
								dn_delaynet *dn, FLOAT_T dt,
								trialparams *p);
void sim_updateneurontraces(FLOAT_T *traces_neu, FLOAT_T *neuronoutputs, IDX_T n,
								FLOAT_T dt, trialparams *p); 
void sim_updatesynapses(FLOAT_T *synapses, FLOAT_T *traces_syn, FLOAT_T *traces_neu, 
							FLOAT_T *neuronoutputs, dn_delaynet *dn, IDX_T *sourceidx,
							FLOAT_T dt, unsigned int numsyn_exc, trialparams *p);
void runstdpmodel(sim_model *m, FLOAT_T *input, size_t inputlen,
					FLOAT_T dur, spikerecord *sr, bool profiling);



#endif
