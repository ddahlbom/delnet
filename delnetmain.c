#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

#ifdef __amd64__
#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
//#include <mpi.h>
#else
#include <mpi.h>
#endif

#include "delnetmpi.h"
#include "simutilsmpi.h"
#include "spkrcd.h"

#define PROFILING 1
#define DN_MAIN_DEBUG 1

#define DN_TRIALTYPE_NEW 0
#define DN_TRIALTYPE_RESUME 1

#define MAX_NAME_LEN 512

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
		//m = su_mpi_izhiblobstdpmodel(model_name, commrank, commsize);
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

	/* set up file names */
	MPI_Barrier(MPI_COMM_WORLD);

	/* set up spike recorder on each rank */
	char srname[MAX_NAME_LEN];
	char rankstr[MAX_NAME_LEN];
	sprintf(rankstr, "_%d_%d", commsize, commrank);
	strcpy(srname, out_name);
	strcat(srname, rankstr);
	strcat(srname, "_spikes.txt");
	spikerecord *sr = sr_init(srname, SPIKE_BLOCK_SIZE);

	/* load input sequence on each rank */
	FILE *infile;
	su_mpi_spike *input_forced;
	long int ninput;
	size_t loadsize;
	if (DN_MAIN_DEBUG) printf("Loading input on process %d\n", commrank);
	if (commrank == 0) {
		/* Read the input file data */
		char infilename[MAX_NAME_LEN];
		strcpy(infilename, in_name);
		strcat(infilename, "_input.bin");
		infile = fopen(infilename, "rb");
		loadsize = fread(&ninput, sizeof(long int), 1, infile);
		if (loadsize != 1) {printf("Failed to load input\n"); exit(-1); }
		input_forced = malloc(sizeof(su_mpi_spike)*ninput);
		loadsize = fread(input_forced, sizeof(su_mpi_spike), ninput, infile);
		if (loadsize != ninput) {printf("Failed to load input\n"); exit(-1); }
		fclose(infile);

		/* Broadcast to all ranks */
		MPI_Bcast( &ninput, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		MPI_Bcast( input_forced, ninput, mpi_spike_type, 0, MPI_COMM_WORLD);
	} else {
		/* Recieve input from rank 0 */
		MPI_Bcast( &ninput, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		input_forced = malloc(sizeof(su_mpi_spike)*ninput);
		MPI_Bcast( input_forced, ninput, mpi_spike_type, 0, MPI_COMM_WORLD);
	}
	if (DN_MAIN_DEBUG) printf("Loaded input on process %d\n", commrank);


	/* run simulation */
	su_mpi_runstdpmodel(m, tp, input_forced, ninput,
						sr, out_name, commrank, commsize, PROFILING);


	/* save resulting model state */
	//su_mpi_savemodel_l(m, outfilename, commsize, commrank);
	su_mpi_savesynapses(m, out_name, commrank, commsize);
	su_mpi_globalsave(m, out_name, commrank, commsize);

	/* Clean up */
	//sr_close(sr);
	char srfinalname[256];
	strcpy(srfinalname, out_name);
	strcat(srfinalname, "_spikes.txt");

	sr_collateandclose(sr, srfinalname, commrank, commsize, mpi_spike_type);

	su_mpi_freemodel_l(m);
	free(input_forced);
	MPI_Finalize();

	return 0;
}
