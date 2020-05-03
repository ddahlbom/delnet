#ifndef SIMKERNELSMPICUDA_H
#define SIMKERNELSMPICUDA_H

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC double sk_mpi_expsampl(double lambda);

EXTERNC void sk_mpi_getinputs(FLOAT_T *neuroninputs, dn_mpi_delaynet *dn, FLOAT_T *synapses);
EXTERNC unsigned int sk_mpi_poisnoise(FLOAT_T *neuroninputs, FLOAT_T *nextrand, FLOAT_T t, 
			 							size_t num_neurons, su_mpi_trialparams *tp);
EXTERNC void sk_mpi_updateneurons(su_mpi_neuron *neurons, FLOAT_T *neuroninputs, IDX_T num_neurons,
			 						su_mpi_trialparams *tp);
EXTERNC unsigned int sk_mpi_checkspiking(su_mpi_neuron *neurons, FLOAT_T *neuronoutputs,
											unsigned int n, FLOAT_T t, spikerecord *sr,
											unsigned int offset);
EXTERNC void sk_mpi_updatesynapsetraces(FLOAT_T *traces_syn, FLOAT_T *spike_pre,
 								dn_mpi_delaynet *dn, FLOAT_T dt,
 								FLOAT_T tau_pre);
EXTERNC void sk_mpi_updatesynapsetraces_cuda(FLOAT_T *traces_syn, FLOAT_T *spike_pre,
											 dn_mpi_delaynet *dn, FLOAT_T dt,
											 FLOAT_T tau_pre);
EXTERNC void sk_mpi_updateneurontraces(FLOAT_T *traces_neu, FLOAT_T *neuronoutputs, IDX_T n,
 								FLOAT_T dt, su_mpi_modelparams *mp);
EXTERNC void sk_mpi_updatesynapses(FLOAT_T *synapses, FLOAT_T *traces_syn, FLOAT_T *traces_neu, 
							FLOAT_T *neuronoutputs, dn_mpi_delaynet *dn, 
							FLOAT_T dt, su_mpi_modelparams *mp);
EXTERNC void sk_mpi_updatesynapses_cuda(FLOAT_T *synapses, FLOAT_T *traces_syn, FLOAT_T *traces_neu, 
										FLOAT_T *neuronoutputs, dn_mpi_delaynet *dn, 
										FLOAT_T dt, FLOAT_T a_pre, FLOAT_T a_post,
										FLOAT_T synmax);
#endif
