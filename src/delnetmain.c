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
long int loadinput(char *in_name, su_mpi_spike **input_forced, MPI_Datatype mpi_spike_type, su_mpi_model_l *m, int commrank)
{
	FILE *infile;
	long int ninput;
	size_t loadsize;
	if (DN_MAIN_DEBUG) printf("Loading input on process %d\n", commrank);
	if (commrank == 0) {
		/* Read the input file data */
		char infilename[MAX_NAME_LEN];
		strcpy(infilename, in_name);
		strcat(infilename, "_input.bin");
		infile = fopen(infilename, "rb");
		checkfileload(infile, infilename);
		loadsize = fread(&ninput, sizeof(long int), 1, infile);
		if (loadsize != 1) {printf("Failed to load input\n"); exit(-1); }
		*input_forced = malloc(sizeof(su_mpi_spike)*ninput);
		loadsize = fread(*input_forced, sizeof(su_mpi_spike), ninput, infile);
		if (loadsize != ninput) {printf("Failed to load input\n"); exit(-1); }
		fclose(infile);
		/* Broadcast to all ranks */
		MPI_Bcast( &ninput, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		MPI_Bcast( *input_forced, ninput, mpi_spike_type, 0, MPI_COMM_WORLD);
	} else {
		/* Recieve input from rank 0 */
		MPI_Bcast( &ninput, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		*input_forced = malloc(sizeof(su_mpi_spike)*ninput);
		MPI_Bcast( *input_forced, ninput, mpi_spike_type, 0, MPI_COMM_WORLD);
	}
	if (DN_MAIN_DEBUG) printf("Loaded input on process %d\n", commrank);

	// Optimize this later -- for now just prune to local input only
	idx_t nlocal=0;
	idx_t i1, i2;
	i1 = m->dn->nodeoffsetglobal;
	i2 = i1 + m->dn->numnodes;

	for (idx_t n=0; n<ninput; n++) 
		if (i1 <= (*input_forced)[n].i && (*input_forced)[n].i < i2) nlocal += 1;

	su_mpi_spike *input_local = 0;
	idx_t c = 0;
	input_local = malloc(sizeof(su_mpi_spike)*nlocal);
	for (idx_t n=0; n<ninput; n++) {
		if (i1 <= (*input_forced)[n].i && (*input_forced)[n].i < i2) {
			input_local[c].i = (*input_forced)[n].i-i1; // ... - i1: put into local indexing basis
			input_local[c].t = (*input_forced)[n].t;
			c++;
		}
	}
	free(*input_forced);
	*input_forced = input_local;

	return nlocal;
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

	char model_name[MAX_NAME_LEN];
	char graph_name[MAX_NAME_LEN];
	char tparams_name[MAX_NAME_LEN];
	strcpy(model_name, in_name);
	strcpy(graph_name, in_name);
	strcpy(tparams_name, in_name);

	int trialtype = atoi(argv[1]);

	/* model create or load */
	if (trialtype == DN_TRIALTYPE_NEW) {
		strcat(model_name, "_mparams.txt");
		strcat(graph_name, "_graph.bin");
		m = su_mpi_izhimodelfromgraph(model_name, graph_name, commrank, commsize);
		if (DN_MAIN_DEBUG) printf("Made model on process %d\n", commrank);
	} else if (trialtype == DN_TRIALTYPE_RESUME) {
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
	su_mpi_spike *input_forced = 0;
	long int ninput = loadinput(in_name, &input_forced, mpi_spike_type, m, commrank);


	/* run simulation */
	if (DN_MAIN_DEBUG) printf("Running simulation on rank %d\n", commrank);
	su_mpi_runstdpmodel(m, tp, input_forced, ninput,
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
	free(input_forced);
	MPI_Finalize();

	return 0;
}
