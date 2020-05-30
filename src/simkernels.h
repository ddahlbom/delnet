#ifndef SIMKERNELSMPI_H
#define SIMKERNELSMPI_H

double sk_mpi_expsampl(double lambda);

void sk_mpi_getinputs(FLOAT_T *neuroninputs, dn_mpi_delaynet *dn, FLOAT_T *synapses);
void sk_mpi_forcedinput( su_mpi_model_l *m, su_mpi_spike *input, size_t ninput, 
						 FLOAT_T *neuroninputs, FLOAT_T t, FLOAT_T dt,
						 double t_max, su_mpi_trialparams *tp,
						 int commrank, int commsize, FILE *inputtimesfile,
						 FLOAT_T *nextrand);
unsigned int sk_mpi_poisnoise(FLOAT_T *neuroninputs, FLOAT_T *nextrand, FLOAT_T t, 
							size_t num_neurons, su_mpi_trialparams *tp);
//void sk_mpi_updateneurons(su_mpi_neuron *neurons, FLOAT_T *neuroninputs, su_mpi_modelparams *mp, //						su_mpi_trialparams *tp);
void sk_mpi_updateneurons(su_mpi_neuron *neurons, FLOAT_T *neuroninputs, IDX_T num_neurons,
						FLOAT_T fs);
unsigned int sk_mpi_checkspiking(su_mpi_neuron *neurons, FLOAT_T *neuronoutputs,
									unsigned int n, FLOAT_T t, spikerecord *sr,
									unsigned int offset, FLOAT_T recordstart,
									FLOAT_T recordstop);
void sk_mpi_updatesynapsetraces(FLOAT_T *traces_syn, FLOAT_T *spike_pre,
								dn_mpi_delaynet *dn, FLOAT_T dt,
								su_mpi_modelparams *mp);
void sk_mpi_updateneurontraces(FLOAT_T *traces_neu, FLOAT_T *neuronoutputs, IDX_T n,
								FLOAT_T dt, su_mpi_modelparams *mp);
void sk_mpi_updatesynapses(FLOAT_T *synapses, FLOAT_T *traces_syn, FLOAT_T *traces_neu, 
							FLOAT_T *neuronoutputs, dn_mpi_delaynet *dn, 
							FLOAT_T dt, su_mpi_modelparams *mp);
#endif