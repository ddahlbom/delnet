#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#include "/usr/include/mpich/mpi.h"
#include <mpi.h>

#include "delnetmpi.h"
#include "simutilsmpi.h"
#include "spkrcd.h"

#define PROFILING 1
#define DEBUG 1



/*************************************************************
 *  Main
 *************************************************************/
int main(int argc, char *argv[])
{
	su_mpi_model_l *m;
	su_mpi_trialparams tp;
	char *infilename;
	char outfilename[256];
	char mpifilename[] = "mpimodel.bin";
	int commsize, commrank;

	/* Init MPI */
	MPI_Init(&argc, &argv);	
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &commrank);
	
	/* rand seed */
	//srand(1);

	/* set parameters */
	if (argc < 4) {
		printf("Need two parameter files (model and trial),\
				input file, and a file name.  Exiting.\n");
		exit(-1);
	}
	m = su_mpi_izhiblobstdpmodel(argv[1], commrank, commsize);
	if (DEBUG) printf("Made model on process %d\n", commrank);

	if (commrank == 0) {
		su_mpi_readtparameters(&tp, argv[2]);
		MPI_Bcast( &tp, sizeof(su_mpi_trialparams), MPI_CHAR, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast( &tp, sizeof(su_mpi_trialparams), MPI_CHAR, 0, MPI_COMM_WORLD);
	}
	if (DEBUG) printf("Loaded trial parameters on process %d\n", commrank);

	infilename = argv[3];
	strcpy(outfilename, argv[4]);

	MPI_Barrier(MPI_COMM_WORLD);
	//m = su_mpi_loadmodel_l(mpifilename); 	/* everyone loads the same model */

	/* set up spike recorder */
	char srname[256];
	char rankstr[256];
	sprintf(rankstr, "_%d", commrank);
	strcpy(srname, outfilename);
	strcat(srname, rankstr);
	strcat(srname, "_spikes.dat");
	spikerecord *sr = sr_init(srname, SPIKE_BLOCK_SIZE);

	/* load input sequence */
	FILE *infile;
	FLOAT_T *input_forced;
	long int N_pat;
	size_t loadsize;


	if (DEBUG) printf("Loading input on process %d\n", commrank);
	if (commrank == 0) {
		infile = fopen(infilename, "rb");

		loadsize = fread(&N_pat, sizeof(long int), 1, infile);
		if (loadsize != 1) {printf("Failed to load input\n"); exit(-1); }

		input_forced = malloc(sizeof(double)*N_pat);
		loadsize = fread(input_forced, sizeof(double), N_pat, infile);
		if (loadsize != N_pat) {printf("Failed to load input\n"); exit(-1); }

		fclose(infile);

		//for (int q=0; q<N_pat; q++) printf("%g\n", input_forced[q]);

		MPI_Bcast( &N_pat, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		MPI_Bcast( input_forced, N_pat, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast( &N_pat, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		input_forced = malloc(sizeof(double)*N_pat);
		MPI_Bcast( input_forced, N_pat, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	if (DEBUG) printf("Loaded input on process %d\n", commrank);


	/* run simulation */
	MPI_Barrier(MPI_COMM_WORLD); 	// Probably unnecessary
	su_mpi_runstdpmodel(m, tp, input_forced, N_pat, sr, commrank, commsize, PROFILING);

	/* save resulting model state */
	char modelfilename[256];
	strcpy(modelfilename, outfilename);
	strcat(modelfilename, rankstr);
	strcat(modelfilename, "_model.bin");
	su_mpi_savemodel_l(m, modelfilename);

	/* Clean up */
	sr_close(sr);
	su_mpi_freemodel_l(m);
	free(input_forced);
	MPI_Finalize();

	return 0;
}
