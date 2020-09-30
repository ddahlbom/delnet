#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <mpi.h>

#include "delnetfixed.h"
#include "simutils.h"
#include "spkrcd.h"

#define PROFILING 1
#define DN_MAIN_DEBUG 0

#define DN_TRIALTYPE_NEW 0
#define DN_TRIALTYPE_RESUME 1

#define MAX_NAME_LEN 512



/*************************************************************
 *  Helper Functions
 *************************************************************/
void getcontributors(su_mpi_model_l *m, idx_t **prev, idx_t *numprev,
					 int commrank, int commsize)
{
	dnf_delaynet *dn = m->dn;
	idx_t numnodes_g = 0;
	int *numnodes_l = malloc(sizeof(int)*commsize);
	int numbuffers_pernode_g = 0; // per node -- a count
	int *numbufferspernode; 
	int *numbufferstotal_perrank = malloc(sizeof(int)*commsize);
	int *nodeoffsets = calloc(commsize, sizeof(int));

	/* Get the number of buffers per nodes globally */
	MPI_Allgather(&dn->numnodes, 1, MPI_UNSIGNED_LONG,
			      numnodes_l, 1, MPI_UNSIGNED_LONG,
				  MPI_COMM_WORLD);
	for (idx_t i=0; i<commsize; i++)
		numnodes_g += numnodes_l[i];
	for (idx_t i=1; i<commsize; i++)
		nodeoffsets[i] = nodeoffsets[i-1] + numnodes_l[i];

	if (commrank == 0) {
		numbufferspernode = malloc(sizeof(int)*numnodes_g);
		MPI_Gatherv(dn->numbuffers, dn->numnodes, MPI_UNSIGNED_LONG, 
					numbufferspernode, numnodes_l, nodeoffsets,
					MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	} else {
		MPI_Gatherv(dn->numbuffers, dn->numnodes, MPI_UNSIGNED_LONG, 
					NULL, NULL, NULL,
					MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	}

	/* Get the buffer sources */
	MPI_Allgather(&dn->numbufferstotal, 1, MPI_UNSIGNED_LONG,
				  numbufferstotal_perrank, 1, MPI_UNSIGNED_LONG,
				  MPI_COMM_WORLD);
				  

	MPI_Gatherv(dn->buffersourcenodes, dn->numbufferstotal, MPI_UNSIGNED_LONG,
				___, ___, ___, 
				MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

	
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

	/* Check number of inputs (needs one) */
	if (argc != 2) {
		printf("No model name given. Exiting...\n");
		exit(-1);
	}

	char *in_name = argv[2];

	/* model create or load */
	m = su_mpi_globalload(in_name, commrank, commsize);
	if (DN_MAIN_DEBUG) printf("Loaded model on process %d\n", commrank);

	/* set up spike recorder on each rank */
	char srname[MAX_NAME_LEN];
	char rankstr[MAX_NAME_LEN];
	sprintf(rankstr, "_%d_%d", commsize, commrank);
	strcpy(srname, in_name);
	strcat(srname, rankstr);
	strcat(srname, "_spikes.txt");
	spikerecord *sr = sr_init(srname, SPIKE_BLOCK_SIZE);

	/* Look of polychronous groups (PGs) */
	if (DN_MAIN_DEBUG) printf("Running simulation on rank %d\n", commrank);
	//su_mpi_runstdpmodel(m, tp, forced_inputs, numinputs,
	//					sr, out_name, commrank, commsize, PROFILING);
		


	/* Clean up */
	char srfinalname[256];
	strcpy(srfinalname, in_name);
	strcat(srfinalname, "_spikes.txt");

	if (DN_MAIN_DEBUG) printf("Saving spikes on rank %d\n", commrank);
	sr_collateandclose(sr, srfinalname, commrank, commsize, mpi_spike_type);

	su_mpi_freemodel_l(m);
	MPI_Finalize();

	return 0;
}
