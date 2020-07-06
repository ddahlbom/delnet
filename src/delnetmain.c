#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <mpi.h>

//#include "delnet.h"
//#include "delnetfixed.h"
#include "simutils.h"
#include "spkrcd.h"

#define PROFILING 1
#define DN_MAIN_DEBUG 1

#define DN_TRIALTYPE_NEW 0
#define DN_TRIALTYPE_RESUME 1

#define MAX_NAME_LEN 512



/*************************************************************
 *  Helper Functions
 *************************************************************/
long int pruneinputtolocal(su_mpi_spike **forced_input, idx_t inputlen, su_mpi_model_l *m) {
	idx_t nlocal=0;
	idx_t i1, i2;
	i1 = m->dn->nodeoffsetglobal;
	i2 = i1 + m->dn->numnodes;

	for (idx_t n=0; n<inputlen; n++) 
		if (i1 <= (*forced_input)[n].i && (*forced_input)[n].i < i2) nlocal += 1;

	su_mpi_spike *input_local = 0;
	idx_t c = 0;
	input_local = malloc(sizeof(su_mpi_spike)*nlocal);
	for (idx_t n=0; n<inputlen; n++) {
		if (i1 <= (*forced_input)[n].i && (*forced_input)[n].i < i2) {
			input_local[c].i = (*forced_input)[n].i-i1; // ... - i1: put into local indexing basis
			input_local[c].t = (*forced_input)[n].t;
			c++;
		}
	}
	free(*forced_input);
	*forced_input = input_local;
	
	return nlocal;
}

long int loadinputs(char *in_name, su_mpi_input **forced_input, MPI_Datatype mpi_spike_type,
				    su_mpi_model_l *m, int commrank)
{
	FILE *infile;
	long int inputlen;
	long int prunedlen;
	long int numinputs;
	size_t loadsize;
	if (DN_MAIN_DEBUG) printf("Loading input on process %d\n", commrank);
	if (commrank == 0) {
		/* Read the input file data */
		char infilename[MAX_NAME_LEN];
		strcpy(infilename, in_name);
		strcat(infilename, "_input.bin");
		infile = fopen(infilename, "rb");
		checkfileload(infile, infilename);

		/* Set up array of input structures */
		loadsize = fread(&numinputs, sizeof(long int), 1, infile);
		if (loadsize != 1) {printf("Failed to load number of inputs\n"); exit(-1); }
		MPI_Bcast(&numinputs, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		*forced_input = malloc(sizeof(su_mpi_input)*numinputs);

		/* Load each input */
		for (idx_t n=0; n<numinputs; n++) {
			/* find out number of spikes in input and allocate */
			loadsize = fread(&inputlen, sizeof(long int), 1, infile);
			if (loadsize != 1) {printf("Failed to load input\n"); exit(-1); }
			(*forced_input)[n].len = inputlen; 
			(*forced_input)[n].spikes = malloc(sizeof(su_mpi_spike)*inputlen);

			/* broadcast input */
			loadsize = fread((*forced_input)[n].spikes, sizeof(su_mpi_spike), inputlen, infile);
			if (loadsize != inputlen) {printf("Failed to load input\n"); exit(-1); }

			/* Broadcast to all ranks */
			MPI_Bcast( &inputlen, 1, MPI_LONG, 0, MPI_COMM_WORLD);
			MPI_Bcast( (*forced_input)[n].spikes, inputlen, mpi_spike_type, 0, MPI_COMM_WORLD);
			prunedlen = pruneinputtolocal(&(*forced_input)[n].spikes, (*forced_input)[n].len, m);
			(*forced_input)[n].len = prunedlen;
		}
		fclose(infile);
	} else {
		/* Get number of inputs and allocate */
		MPI_Bcast(&numinputs, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		*forced_input = malloc(sizeof(su_mpi_input)*numinputs);

		for (idx_t n=0; n<numinputs; n++) {
			/* Recieve input from rank 0 */
			MPI_Bcast( &inputlen, 1, MPI_LONG, 0, MPI_COMM_WORLD);
			(*forced_input)[n].len = inputlen;
			(*forced_input)[n].spikes = malloc(sizeof(su_mpi_spike)*inputlen);
			MPI_Bcast( (*forced_input)[n].spikes, inputlen, mpi_spike_type, 0, MPI_COMM_WORLD);
			prunedlen = pruneinputtolocal(&(*forced_input)[n].spikes, (*forced_input)[n].len, m);
			(*forced_input)[n].len = prunedlen;
		}
	}
	if (DN_MAIN_DEBUG) printf("Loaded input on process %d\n", commrank);

	return numinputs;
}

void freeinputs(su_mpi_input **forced_input, idx_t numinputs) {
	for (idx_t n=0; n<numinputs; n++) 
		free((*forced_input)[n].spikes);
	free(*forced_input);
}


/*************************************************************
 *  Main
 *************************************************************/
int main(int argc, char *argv[])
{
	su_mpi_model_l *m;
	su_mpi_trialparams tp;
	int commsize, commrank;

	/* Init MPI */
	MPI_Init(&argc, &argv);	
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &commrank);

	/* Set up MPI Datatype for spikes */
	MPI_Datatype mpi_spike_type = sr_commitmpispiketype();

	/* Check number of inputs (needs three) */
	if (argc != 4) {
		printf("Input: Trial type (0 or 1), input name, output name.\n");
		exit(-1);
	}

	char *in_name = argv[2];
	char *out_name = argv[3];

	char tparams_name[MAX_NAME_LEN];
	//strcpy(model_name, in_name);
	//strcpy(graph_name, in_name);
	strcpy(tparams_name, in_name);

	int trialtype = atoi(argv[1]);

	/* model create or load */
	if (trialtype == DN_TRIALTYPE_NEW) {
		//strcat(model_name, "_mparams.txt");
		//strcat(graph_name, "_graph.bin");
		m = su_mpi_izhimodelfromgraph(in_name, commrank, commsize);
		if (DN_MAIN_DEBUG) printf("Made model on process %d\n", commrank);
	} else if (trialtype == DN_TRIALTYPE_RESUME) {
		char model_name[MAX_NAME_LEN];
		strcpy(model_name, in_name);
		strcat(model_name, "_model.bin");
		m = su_mpi_globalload(model_name, commrank, commsize);
		if (DN_MAIN_DEBUG) printf("Loaded model on process %d\n", commrank);
	} else {
		printf("First argument must be 0 or 1. Exiting.\n");
		exit(-1);
	}

	/* process trial parameters */
	if (commrank == 0) {
		strcat(tparams_name, "_tparams.txt");
		su_mpi_readtparameters(&tp, tparams_name);
		MPI_Bcast( &tp, sizeof(su_mpi_trialparams), MPI_CHAR, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast( &tp, sizeof(su_mpi_trialparams), MPI_CHAR, 0, MPI_COMM_WORLD);
	}
	if (DN_MAIN_DEBUG) printf("Loaded trial parameters on process %d\n", commrank);

	/* set up spike recorder on each rank */
	char srname[MAX_NAME_LEN];
	char rankstr[MAX_NAME_LEN];
	sprintf(rankstr, "_%d_%d", commsize, commrank);
	strcpy(srname, out_name);
	strcat(srname, rankstr);
	strcat(srname, "_spikes.txt");
	spikerecord *sr = sr_init(srname, SPIKE_BLOCK_SIZE);

	/* load input sequence on each rank */
	su_mpi_input *forced_inputs = 0;
	long int numinputs = loadinputs(in_name, &forced_inputs, mpi_spike_type, m, commrank);


	/* run simulation */
	if (DN_MAIN_DEBUG) printf("Running simulation on rank %d\n", commrank);
	su_mpi_runstdpmodel(m, tp, forced_inputs, numinputs,
						sr, out_name, commrank, commsize, PROFILING);


	/* save resulting model state */
	if (DN_MAIN_DEBUG) printf("Saving synapses on rank %d\n", commrank);
	su_mpi_savesynapses(m, out_name, commrank, commsize);
	if (DN_MAIN_DEBUG) printf("Saving model state on rank %d\n", commrank);
	su_mpi_globalsave(m, out_name, commrank, commsize);

	/* Clean up */
	char srfinalname[256];
	strcpy(srfinalname, out_name);
	strcat(srfinalname, "_spikes.txt");

	if (DN_MAIN_DEBUG) printf("Saving spikes on rank %d\n", commrank);
	sr_collateandclose(sr, srfinalname, commrank, commsize, mpi_spike_type);

	su_mpi_freemodel_l(m);
	freeinputs(&forced_inputs, numinputs);
	MPI_Finalize();

	return 0;
}
