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

typedef struct modelparams_s {
	double fs; 				// <- both
	double num_neurons; 	// <- structural
	double p_contact; 		// <- structural
	double p_exc; 			// <- structural
	double maxdelay; 		// <- structrual
	double tau_pre; 		// <- structural
	double tau_post; 		// <- structural
	double a_pre; 			// <- structural
	double a_post; 			// <- structural
	double synmax; 			// <- structural
	double w_exc; 			// <- structural
	double w_inh; 			// <- structural
} modelparams;

typedef struct trialparams_s {
	double fs; 				// <- both
	double dur; 			// <- trial
	double lambda; 			// <- trial
	double randspikesize; 	// <- trial
	bool randinput;
	bool inhibition;
	IDX_T numinputs;
	IDX_T *inputidcs;
} trialparams;

typedef struct sim_model_s {
	IDX_T numinputneurons;
	IDX_T numsyn_exc;
	modelparams p;
	dn_delaynet *dn;
	neuron  *neurons;
	FLOAT_T *traces_neu;
	FLOAT_T *traces_syn;
	FLOAT_T *synapses;
} sim_model;

/*************************************************************
 *  Function Declarations
 *************************************************************/

/* parameter bookkeeping */
void setdefaultmparams(modelparams *p);
void readmparameters(modelparams *p, char *filename);
void printmparameters(modelparams p);
void analyzeconnectivity(unsigned int *g, unsigned int n,
							unsigned int n_exc, FLOAT_T fs);

/* graph generation*/
unsigned int *iblobgraph(modelparams *p);

/* neuron info */
void neuron_set(neuron *n, FLOAT_T v, FLOAT_T u, FLOAT_T a, FLOAT_T d);
void neuronupdate_rk4(FLOAT_T *v, FLOAT_T *u, FLOAT_T input, FLOAT_T a, FLOAT_T h);

/* other utility functions */
double expsampl(double lambda);

/* save and load and free models */
void sim_savemodel(sim_model *m, char *filename);
sim_model *sim_loadmodel(char *filename);
void sim_freemodel(sim_model *m);

/* kernels */
void sim_getinputs(FLOAT_T *neuroninputs, dn_delaynet *dn, FLOAT_T *synapses);
unsigned int sim_poisnoise(FLOAT_T *neuroninputs, FLOAT_T *nextrand, FLOAT_T t, 
								size_t num_neurons, trialparams *tp);
void sim_updateneurons(neuron *neurons, FLOAT_T *neuroninputs,
						modelparams *mp, trialparams *p);
unsigned int sim_checkspiking(neuron *neurons, FLOAT_T *neuronoutputs,
								unsigned int n, FLOAT_T t, spikerecord *sr);
void sim_updatesynapsetraces(FLOAT_T *traces_syn, FLOAT_T *spike_pre,
								dn_delaynet *dn, FLOAT_T dt,
								modelparams *mp);
void sim_updateneurontraces(FLOAT_T *traces_neu, FLOAT_T *neuronoutputs, IDX_T n,
								FLOAT_T dt, modelparams *p); 
void sim_updatesynapses(FLOAT_T *synapses, FLOAT_T *traces_syn, FLOAT_T *traces_neu, 
							FLOAT_T *neuronoutputs, dn_delaynet *dn, IDX_T *sourceidx,
							FLOAT_T dt, unsigned int numsyn_exc, modelparams *p);
void runstdpmodel(sim_model *m, trialparams tp, FLOAT_T *input, size_t inputlen,
					spikerecord *sr, bool profiling);



#endif
