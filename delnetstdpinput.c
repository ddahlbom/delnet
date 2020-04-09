#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "delnet.h"
#include "spkrcd.h"
#include "simutils.h"

#define PROFILING 1


/*************************************************************
 *  Main
 *************************************************************/
int main(int argc, char *argv[])
{
	su_model *m;
	su_trialparams tp;
	char *infilename;
	char outfilename[256];

	srand(1);

	/* set parameters */
	if (argc < 4) {
		printf("Need two parameter files (model and trial),\
				input file, and a file name.  Exiting.\n");
		exit(-1);
	} else {
		m = su_izhiblobstdpmodel(argv[1]);
		su_readtparameters(&tp, argv[2]);
		infilename = argv[3];
		strcpy(outfilename, argv[4]);
	}

	/* set up spike recorder */
	char srname[256];
	strcpy(srname, outfilename);
	strcat(srname, "_spikes.dat");
	spikerecord *sr = sr_init(srname, SPIKE_BLOCK_SIZE);

	/* load input sequence */
	FILE *infile;
	FLOAT_T *input_forced;
	long int N_pat;
	size_t loadsize;

	infile = fopen(infilename, "rb");
	loadsize = fread(&N_pat, sizeof(long int), 1, infile);
	if (loadsize != 1) {printf("Failed to load input\n"); exit(-1); }
	input_forced = malloc(sizeof(double)*N_pat);
	loadsize = fread(input_forced, sizeof(double), N_pat, infile);
	if (loadsize != N_pat) {printf("Failed to load input\n"); exit(-1); }
	fclose(infile);

	/* run simulation */
	su_runstdpmodel(m, tp, input_forced, N_pat, sr, PROFILING);

	/* save resulting model state */
	char modelfilename[256];
	strcpy(modelfilename, outfilename);
	strcat(modelfilename, "_model.bin");
	su_savemodel(m, modelfilename);

	/* Clean up */
	sr_close(sr);
	su_freemodel(m);
	free(input_forced);

	return 0;
}
