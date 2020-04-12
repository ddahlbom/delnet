#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "/usr/include/mpich/mpi.h"
//#include <mpi.h>

#include "delnetmpi.h"
#include "simutilsmpi.h"
#include "spkrcd.h"

#define PROFILING 1



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
	srand(1);

	/* set parameters */
	if (argc < 4) {
		printf("Need two parameter files (model and trial),\
				input file, and a file name.  Exiting.\n");
		exit(-1);
	} else {
		if (commrank==0) {
			m = su_mpi_izhiblobstdpmodel(argv[1], commrank, commsize);
			su_mpi_savemodel_l(m, "mpimodel.bin");
			su_mpi_freemodel_l(m);
		}
		su_mpi_readtparameters(&tp, argv[2]);
		infilename = argv[3];
		strcpy(outfilename, argv[4]);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	m = su_mpi_loadmodel_l(mpifilename); 	/* everyone loads the same model */

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

	infile = fopen(infilename, "rb");
	loadsize = fread(&N_pat, sizeof(long int), 1, infile);
	if (loadsize != 1) {printf("Failed to load input\n"); exit(-1); }
	input_forced = malloc(sizeof(double)*N_pat);
	loadsize = fread(input_forced, sizeof(double), N_pat, infile);
	if (loadsize != N_pat) {printf("Failed to load input\n"); exit(-1); }
	fclose(infile);

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
