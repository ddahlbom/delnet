#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

#ifdef __amd64__
#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif

#include "delnetmpi.h"
#include "simutilsmpi.h"
#include "spkrcd.h"

#define PROFILING 1
#define DEBUG 0

/*************************************************************
 *  Main
 *************************************************************/
int main(int argc, char *argv[])
{
	su_mpi_model_l *m;
	su_mpi_trialparams tp;
	char *infilename;
	char outfilename[256];
	int commsize, commrank;

	/* Init MPI */
	MPI_Init(&argc, &argv);	
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &commrank);

	/* Set up MPI Datatype for spikes */
	const int 		nitems = 2;
	int 			blocklengths[2] = {1, 1};
	MPI_Datatype	types[2] = {MPI_UNSIGNED, MPI_DOUBLE};
	MPI_Datatype	mpi_spike_type;
	MPI_Aint  		offsets[2];

	offsets[0] = offsetof(su_mpi_spike, i);
	offsets[1] = offsetof(su_mpi_spike, t);
	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_spike_type);
	MPI_Type_commit(&mpi_spike_type);
	
	/* set parameters from (dumb) CLI */
	if (argc < 6) {
		printf("Need trial type (0 - new model, 1 - loadmodel), model\
				filename (parameter file or existing model), trial\
				parameter file,\
				input file, and a file name.  Exiting.\n");
		exit(-1);
	}


	/* model create or load */
	if (atoi(argv[1]) == 0) {
		m = su_mpi_izhiblobstdpmodel(argv[2], commrank, commsize);
		if (DEBUG) printf("Made model on process %d\n", commrank);
	} else if (atoi(argv[1]) == 1) {
		//m = su_mpi_loadmodel_l(argv[2], commrank, commsize);
		m = su_mpi_globalload(argv[2], commrank, commsize);
		if (DEBUG) printf("Loaded model on process %d\n", commrank);
	} else {
		printf("First argument must be 0 or 1. Exiting.\n");
		exit(-1);
	}

	/* process trial parameters */
	if (commrank == 0) {
		su_mpi_readtparameters(&tp, argv[3]);
		MPI_Bcast( &tp, sizeof(su_mpi_trialparams), MPI_CHAR, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast( &tp, sizeof(su_mpi_trialparams), MPI_CHAR, 0, MPI_COMM_WORLD);
	}
	if (DEBUG) printf("Loaded trial parameters on process %d\n", commrank);

	/* set up file names */
	char *trialname = argv[5];
	infilename = argv[4];
	strcpy(outfilename, trialname);

	MPI_Barrier(MPI_COMM_WORLD);
	//m = su_mpi_loadmodel_l(mpifilename); 	/* everyone loads the same model */

	/* set up spike recorder on each rank */
	char srname[256];
	char rankstr[256];
	sprintf(rankstr, "_%d_%d", commsize, commrank);
	strcpy(srname, outfilename);
	strcat(srname, rankstr);
	strcat(srname, "_spikes.txt");
	spikerecord *sr = sr_init(srname, SPIKE_BLOCK_SIZE);

	/* load input sequence on each rank */
	FILE *infile;
	su_mpi_spike *input_forced;
	long int ninput;
	size_t loadsize;
	if (DEBUG) printf("Loading input on process %d\n", commrank);
	if (commrank == 0) {
		/* Read the input file data */
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
	if (DEBUG) printf("Loaded input on process %d\n", commrank);


	/* run simulation */
	su_mpi_runstdpmodel(m, tp, input_forced, ninput,
						sr, trialname, commrank, commsize, PROFILING);


	/* save resulting model state */
	//su_mpi_savemodel_l(m, outfilename, commsize, commrank);
	su_mpi_savesynapses(m, trialname, commrank, commsize);
	su_mpi_globalsave(m, trialname, commrank, commsize);

	/* Clean up */
	//sr_close(sr);
	char srfinalname[256];
	strcpy(srfinalname, trialname);
	strcat(srfinalname, "_spikes.txt");

	sr_collateandclose(sr, srfinalname, commrank, commsize);

	su_mpi_freemodel_l(m);
	free(input_forced);
	MPI_Finalize();

	return 0;
}
