#include <math.h>
#include <stdlib.h>

#include "delnetmpi.h"
#include "simutilsmpi.h"
#include "simkernelsmpi.h"

#ifdef __amd64__
#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif

/* -------------------- Random Sampling -------------------- */
double sk_mpi_expsampl(double lambda)
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

void sk_mpi_forcedinput( su_mpi_model_l *m, su_mpi_spike *input, size_t ninput, 
						 FLOAT_T *neuroninputs, FLOAT_T t, FLOAT_T dt,
						 double t_max, su_mpi_trialparams *tp,
						 int commrank, int commsize, FILE *inputtimesfile )
{
	static double t_local = 0.0;
	static double nextinputtime = 0.0;
	static bool waiting = true;

	if (tp->inputmode == INPUT_MODE_PERIODIC) {
		if ( t_local == 0.0 && commrank==0 ) fprintf(inputtimesfile, "%f\n", t);
		for (size_t k=0; k < ninput; k++) {
			if (m->nodeoffset <= (input[k].i - 1) && (input[k].i - 1) < m->nodeoffset + m->maxnode) {
				if (t_local <= input[k].t && input[k].t < t_local + dt) 
					neuroninputs[input[k].i - 1 - m->nodeoffset] += tp->inputweight; 
			}
		}
		t_local += dt;
		if (t_local > t_max) t_local = 0; 
	}
	else if (tp->inputmode == INPUT_MODE_POISSON) {
		if (waiting) {
			if (t >= nextinputtime) {
				waiting = false;
				if (commrank == 0) fprintf(inputtimesfile, "%f\n", t);
			}
		}
		if (!waiting) {
			for (size_t k=0; k < ninput; k++) {
				if (m->nodeoffset <= (input[k].i - 1) && (input[k].i - 1) < m->nodeoffset + m->maxnode) {
					if (t_local <= input[k].t && input[k].t < t_local + dt) 
						neuroninputs[input[k].i - 1 - m->nodeoffset] += tp->inputweight; 
				}
			}
			t_local += dt;
			if (t_local > t_max) {
				waiting = true;
				t_local = 0.0;
				if (commrank==0) {
					nextinputtime = t + sk_mpi_expsampl(tp->lambdainput);
					MPI_Bcast(&nextinputtime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				} else {
					MPI_Bcast(&nextinputtime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				}
			}
		}
	}
}

unsigned int sk_mpi_poisnoise(FLOAT_T *neuroninputs, FLOAT_T *nextrand, FLOAT_T t, 
							size_t num_neurons, su_mpi_trialparams *tp)
{
	//  random input
	unsigned int num = 0, k;
	for (k=0; k<num_neurons; k++) {
		if (nextrand[k] < t) {
			neuroninputs[k] += tp->randspikesize;
			nextrand[k] += sk_mpi_expsampl(tp->lambda);
			num += 1;
		}
	}
	return num;
}

void sk_mpi_updateneurons(su_mpi_neuron *neurons, FLOAT_T *neuroninputs, IDX_T num_neurons,
						FLOAT_T fs)
{
	size_t k;
	for (k=0; k<num_neurons; k++) {
		neuronupdate_rk4(&neurons[k].v, &neurons[k].u, neuroninputs[k],
							neurons[k].a, 1000.0/fs);
	}
}

unsigned int sk_mpi_checkspiking(su_mpi_neuron *neurons, FLOAT_T *neuronoutputs,
									unsigned int n, FLOAT_T t, spikerecord *sr,
									unsigned int offset, FLOAT_T recordstart,
									FLOAT_T recordstop)
{
	size_t k;
	unsigned int numspikes=0;
	for (k=0; k<n; k++) {
		neuronoutputs[k] = 0.0;
		if (neurons[k].v >= 30.0) {
			if( recordstart <= t && t < recordstop)
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
