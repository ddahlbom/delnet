#include <math.h>
#include <stdlib.h>

#include "delnetmpi.h"
#include "simutilsmpi.h"
#include "simkernelsmpi.h"


/* -------------------- Random Sampling -------------------- */

/*
 * Samples from an exponential distribution. For generating 
 * Poissonian noise.
 */
double sk_mpi_expsampl(double lambda)
{
	return -log( (((double) rand()) / ((double) RAND_MAX + 1.0)))/lambda;
}



/* -------------------- Neuron Equations -------------------- */

/*
 * First function for Runge-Kutta method. This is the Izhikevich
 * "simple" model, voltage variable.
 */
static inline FLOAT_T f1(FLOAT_T v, FLOAT_T u, FLOAT_T input) {
	return (0.04*v + 5.0)*v + 140.0 - u + input;
}

/*
 * Second function for Runge-Kutta method. This is the Izhikevich
 * "simple" model, recover variable.
 */
static inline FLOAT_T f2(FLOAT_T v, FLOAT_T u, FLOAT_T a) {
	return a*(0.2*v - u);
}

/*
 * Update neuron state using 4th order Runge-Kutta
 */
void neuronupdate_rk4(FLOAT_T *v, FLOAT_T *u, FLOAT_T input, FLOAT_T a, FLOAT_T h) {
	FLOAT_T K1, K2, K3, K4, L1, L2, L3, L4, half_h, sixth_h;

	half_h = h*0.5;
	sixth_h = h/6.0;
	
	K1 = f1(*v, *u, 0.0);
	L1 = f2(*v, *u, a);

	K2 = f1(*v + half_h*K1, *u + half_h*L1, 0.0); 
	L2 = f2(*v + half_h*K1, *u + half_h*L1, a);

	K3 = f1(*v + half_h*K2, *u + half_h*L2, 0.0);
	L3 = f2(*v + half_h*K2, *u + half_h*L2, a);

	K4 = f1(*v + h*K3, *u + h*L3, 0.0);
	L4 = f2(*v + h*K3, *u + h*L3, a);

	*v = *v + sixth_h * (K1 + 2*K2 + 2*K3 + K4) + input;
	*u = *u + sixth_h * (L1 + 2*L2 + 2*L3 + L4); 
}


/*-------------------- Kernels -------------------- */

/*
 * Takes the neuron inputs, multiples them by appropriate
 * synaptic weight, sums, and returns result.
 */
void sk_mpi_getinputs(FLOAT_T *neuroninputs, dn_mpi_delaynet *dn, FLOAT_T *synapses)
{
	size_t k,j;
	FLOAT_T *delayoutputs;
	for (k=0; k<dn->num_nodes_l; k++) {
		// get inputs to neuron (outputs of delaylines)
		neuroninputs[k] = 0.0;
		delayoutputs = dn_mpi_getinputaddress(k,dn); //dn->outputs
		for (j=0; j < dn->nodes[k].num_in; j++) {
			neuroninputs[k] += delayoutputs[j] * synapses[ dn->nodes[k].idx_outbuf+j ];
		}
	}
}


/*
 * An update function for generating Poissonian input noise.
 */
unsigned int sk_mpi_poisnoise(FLOAT_T *neuroninputs, FLOAT_T *nextrand, FLOAT_T t, 
							size_t num_neurons, su_mpi_trialparams *tp)
{
	unsigned int num = 0, k;
	for (k=0; k<num_neurons; k++) {
		if (nextrand[k] < t) {
			//neuroninputs[k] += p->randspikesize * (p->fs/1000); 
			neuroninputs[k] += tp->randspikesize;
			nextrand[k] += sk_mpi_expsampl(tp->lambda);
			num += 1;
		}
	}
	return num;
}


/*
 * Function for updating all neurons (calls RK update function above for all
 * nodes).
 */
void sk_mpi_updateneurons(su_mpi_neuron *neurons, FLOAT_T *neuroninputs, IDX_T num_neurons,
						su_mpi_trialparams *tp)
{
	size_t k;
	for (k=0; k<num_neurons; k++) {
		neuronupdate_rk4(&neurons[k].v, &neurons[k].u, neuroninputs[k],
							neurons[k].a, 1000.0/tp->fs);
	}
}

unsigned int sk_mpi_checkspiking(su_mpi_neuron *neurons, FLOAT_T *neuronoutputs,
									unsigned int n, FLOAT_T t, spikerecord *sr,
									unsigned int offset)
{
	size_t k;
	unsigned int numspikes=0;
	for (k=0; k<n; k++) {
		neuronoutputs[k] = 0.0;
		if (neurons[k].v >= 30.0) {
			sr_save_spike(sr, k+offset, t);
			neuronoutputs[k] = 1.0;
			neurons[k].v = -65.0;
			neurons[k].u += neurons[k].d;
			numspikes += 1;
		}
	}
	return numspikes;
}

void sk_mpi_updatesynapsetraces(FLOAT_T *traces_syn, FLOAT_T *spike_pre,
								dn_mpi_delaynet *dn, FLOAT_T dt,
								su_mpi_modelparams *mp)
{
	size_t k, j;
	FLOAT_T *neuroninputs;

	for (k=0; k<dn->num_nodes_l; k++) {
		for (j=0; j < dn->nodes[k].num_in; j++) {
			neuroninputs = dn_mpi_getinputaddress(k,dn);
			spike_pre[dn->nodes[k].idx_outbuf +j] = neuroninputs[j];
			traces_syn[dn->nodes[k].idx_outbuf +j] = traces_syn[dn->nodes[k].idx_outbuf +j]*(1.0 - (dt/mp->tau_pre)) +
				spike_pre[dn->nodes[k].idx_outbuf +j];
		}
	}
}


void sk_mpi_updateneurontraces(FLOAT_T *traces_neu, FLOAT_T *neuronoutputs, IDX_T n,
								FLOAT_T dt, su_mpi_modelparams *mp) 
{
	size_t k;
	for (k=0; k<n; k++) { 		
		traces_neu[k] = traces_neu[k]*(1.0 - (dt/mp->tau_post)) + neuronoutputs[k];
	}
}

void sk_mpi_updatesynapses(FLOAT_T *synapses, FLOAT_T *traces_syn, FLOAT_T *traces_neu, 
							FLOAT_T *neuronoutputs, dn_mpi_delaynet *dn, 
							FLOAT_T dt, su_mpi_modelparams *mp)
{
	size_t k, j;
	FLOAT_T *synapseoutputs = dn->outputs;
	for (k=0; k<dn->num_nodes_l; k++) 
	for (j=0; j < dn->nodes[k].num_in; j++) {
		// only update excitatory synapses
		if (synapses[dn->nodes[k].idx_outbuf+j] > 0) {
			synapses[dn->nodes[k].idx_outbuf+j] = synapses[dn->nodes[k].idx_outbuf+j] +
					dt * (mp->a_post * traces_syn[dn->nodes[k].idx_outbuf+j] * neuronoutputs[k] -
						  mp->a_pre * traces_neu[k] * synapseoutputs[dn->nodes[k].idx_outbuf+j]);
			// clamp value	
			synapses[dn->nodes[k].idx_outbuf+j] = synapses[dn->nodes[k].idx_outbuf+j] < 0.0 ? 
										0.0 : synapses[dn->nodes[k].idx_outbuf+j];
			synapses[dn->nodes[k].idx_outbuf+j] = synapses[dn->nodes[k].idx_outbuf+j] > mp->synmax ?
										mp->synmax : synapses[dn->nodes[k].idx_outbuf+j];
		}
	}
	
}
