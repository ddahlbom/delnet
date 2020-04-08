#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "delnet.h"
#include "spkrcd.h"
#include "paramutils.h"
#include "simutils.h"


/*************************************************************
 *  Macros
 *************************************************************/
#define PROFILING 1

/*************************************************************
 *  Globals
 *************************************************************/
//FLOAT_T g_v_default = -65.0;
//FLOAT_T g_u_default = -13.0;
//
//FLOAT_T g_a_exc  = 0.02;
//FLOAT_T g_d_exc  = 8.0;
//
//FLOAT_T g_a_inh  = 0.1;
//FLOAT_T g_d_inh  = 2.0;

/*************************************************************
 *  Main
 *************************************************************/

int main(int argc, char *argv[])
{
	//modelparams p;
	unsigned int i, k;
	unsigned int n, n_exc;
	//unsigned int *g;
	//dn_delaynet *dn;
	spikerecord *sr = sr_init("delnetstdpinput.dat", SPIKE_BLOCK_SIZE);

	srand(1);

	///* set parameters */
	//if (argc < 2) {
	//	printf("No parameter file given.  Using defaults. \n");
	//	setdefaultmparams(&p);
	//} else {
	//	readmparameters(&p, argv[1]);
	//}

	///* derived parameters */
	//n = p.num_neurons;
	//n_exc = (unsigned int) ( (double) n * p.p_exc);

	///* print parameters */
	//printmparameters(p);

	///* set up graph */
	//g = iblobgraph(&p);

	///* analyze connectivity -- sanity check*/
	//analyzeconnectivity(g, n, n_exc, p.fs);

	///* generate delay network */
	//dn = dn_delnetfromgraph(g, n);

	///* initialize neuron and synapse state  */
	//neuron *neurons = malloc(sizeof(neuron)*n);
	//FLOAT_T *traces_neu = calloc(n, sizeof(FLOAT_T));
	//FLOAT_T *traces_syn; 	// pack this
	//FLOAT_T *synapses; 		// for speed?
	//IDX_T numsyn_tot=0, numsyn_exc;

	//for (i=0; i<n_exc; i++) {
	//	neuron_set(&neurons[i], g_v_default, g_u_default, g_a_exc, g_d_exc);
	//	numsyn_tot += dn->nodes[i].num_in;
	//}
	//numsyn_exc = numsyn_tot;
	//for (i=n_exc; i<n; i++) {
	//	neuron_set(&neurons[i], g_v_default, g_u_default, g_a_inh, g_d_inh);
	//	numsyn_tot += dn->nodes[i].num_in;
	//}
	//traces_syn = calloc(numsyn_tot, sizeof(FLOAT_T));		
	//synapses = calloc(numsyn_tot, sizeof(FLOAT_T));
	//
	///* initialize synapse weights */
	//for (i=0; i < numsyn_exc; i++)
	//	synapses[dn->destidx[i]] = p.w_exc;
	//for (; i < numsyn_tot; i++)
	//	synapses[dn->destidx[i]] = p.w_inh;

	sim_model *m2 = izhiblobstdpmodel(argv[1]);
	///* Generate an input sequence to repeat and save */
	FLOAT_T dur_pat = 0.100;
	FLOAT_T mag_pat = 20.0;
	size_t N_pat = (size_t) (dur_pat * m2->p.fs);
	FLOAT_T f_pat = 10.0;
	size_t dn_pat = (size_t) m2->p.fs/ (size_t) f_pat;
	FLOAT_T *input_forced = calloc(N_pat, sizeof(FLOAT_T));
	size_t numinputneurons = 100;
	FILE *infile;
	infile = fopen("forcedinput.dat", "w");
	for (i=0; i<N_pat; i++) {
		input_forced[i] = i % dn_pat == 0 ? mag_pat : 0.0;
		fprintf(infile, "%g\n", input_forced[i]);
	}
	fclose(infile);

	///* run simulation */
	//sim_model m = { numinputneurons, numsyn_exc, p, dn, neurons, 
	//				traces_neu, traces_syn, synapses };
	//sim_savemodel(&m, "model.bin");	
	//dn_freedelnet(dn);
	//free(neurons);
	//free(traces_neu);
	//free(traces_syn);
	//free(synapses);

	//printf("----------------------------------------\n");
	//sim_model *m2 = sim_loadmodel("model.bin");
	trialparams tp = { .fs  = m2->p.fs, 
					   .dur = 2.5,
					   .lambda = 3.0,
					   .randspikesize = 20.0,
					   .randinput=true,
					   .inhibition=true,
					   .numinputs=100,
					   .inputidcs=NULL };

	sim_runstdpmodel(m2, tp, input_forced, N_pat, sr, PROFILING);

	/* save synapse weights */
	//FILE *f;
	//f = fopen("synapses.dat", "w");
	//for (k=0; k<numsyn_tot; k++) {
	//	fprintf(f, "%g\n", m2->synapses[m2->dn->destidx[k]]);
	//}
	//fclose(f);

	/* Clean up */
	sr_close(sr);
	sim_freemodel(m2);

	//free(g);
	free(input_forced);

	return 0;
}
