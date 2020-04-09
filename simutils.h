#ifndef SIMUTILS_H
#define SIMUTILS_H

#include <stdbool.h>

#include "spkrcd.h"
#include "delnet.h"

/*************************************************************
 *  Structs
 *************************************************************/
typedef struct su_neuron_s {
	FLOAT_T v;
	FLOAT_T u;
	FLOAT_T a;
	FLOAT_T d;
} su_neuron;

typedef struct su_modelparams_s {
	double fs; 				
	double num_neurons; 	
	double p_contact; 		
	double p_exc; 			
	double maxdelay; 		
	double tau_pre; 		
	double tau_post; 		
	double a_pre; 			
	double a_post; 			
	double synmax; 			
	double w_exc; 			
	double w_inh; 			
} su_modelparams;

typedef struct su_trialparams_s {
	double fs; 				
	double dur; 			
	double lambda; 			
	double randspikesize; 	
	bool randinput;
	bool inhibition;
	IDX_T numinputs;
	IDX_T *inputidcs;
} su_trialparams;

typedef struct su_model_s {
	IDX_T numinputneurons;
	IDX_T numsyn_exc;
	su_modelparams p;
	dn_delaynet *dn;
	su_neuron  *neurons;
	FLOAT_T *traces_neu;
	FLOAT_T *traces_syn;
	FLOAT_T *synapses;
} su_model;

/*************************************************************
 *  Function Declarations
 *************************************************************/

/* parameter bookkeeping */
void su_setdefaultmparams(su_modelparams *p);
void su_readmparameters(su_modelparams *p, char *filename);
void su_readtparameters(su_trialparams *p, char *filename);
void su_printmparameters(su_modelparams p);
void su_analyzeconnectivity(unsigned int *g, unsigned int n,
							unsigned int n_exc, FLOAT_T fs);

/* graph generation*/
unsigned int *su_iblobgraph(su_modelparams *p);

/* neuron info */
void su_neuronset(su_neuron *n, FLOAT_T v, FLOAT_T u, FLOAT_T a, FLOAT_T d);

/* save and load and free models */
void su_savemodel(su_model *m, char *filename);
su_model *su_loadmodel(char *filename);
void su_freemodel(su_model *m);

/* generate model */
su_model *su_izhiblobstdpmodel(char *mparamfilename);

/* run simulations */
void su_runstdpmodel(su_model *m, su_trialparams tp, FLOAT_T *input, size_t inputlen,
					spikerecord *sr, bool profiling);


#endif
