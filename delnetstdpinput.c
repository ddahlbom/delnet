#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "delnet.h"
#include "spkrcd.h"
//#include "paramutils.h"
#include "simutils.h"

#define PROFILING 1


/*************************************************************
 *  Main
 *************************************************************/
int main(int argc, char *argv[])
{
	unsigned int i;
	spikerecord *sr = sr_init("delnetstdpinput.dat", SPIKE_BLOCK_SIZE);
	sim_model *m;
	trialparams tp;

	srand(1);

	/* set parameters */
	if (argc < 3) {
		printf("Need two parameter files (model and trial).  Exiting.\n");
		exit(-1);
	} else {
		m = izhiblobstdpmodel(argv[1]);
		readtparameters(&tp, argv[2]);
	}

	/* Generate an input sequence to repeat and save */
	FLOAT_T dur_pat = 0.100;
	FLOAT_T mag_pat = 20.0;
	size_t N_pat = (size_t) (dur_pat * m->p.fs);
	FLOAT_T f_pat = 10.0;
	size_t dn_pat = (size_t) m->p.fs/ (size_t) f_pat; 	// step between spikes
	FLOAT_T *input_forced = calloc(N_pat, sizeof(FLOAT_T));
	FILE *infile;
	infile = fopen("forcedinput.dat", "w");
	for (i=0; i<N_pat; i++) {
		input_forced[i] = i % dn_pat == 0 ? mag_pat : 0.0;
		fprintf(infile, "%g\n", input_forced[i]);
	}
	fclose(infile);

	/* run simulation */
	sim_runstdpmodel(m, tp, input_forced, N_pat, sr, PROFILING);

	/* save resulting model state */
	sim_savemodel(m, "modelposttrial.dat");

	/* Clean up */
	sr_close(sr);
	sim_freemodel(m);
	free(input_forced);

	return 0;
}
