#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <mpi.h>

#include "delnet.h"
#include "simutils.h"
#include "spkrcd.h"

#define PROFILING 1
#define PG_MAIN_DEBUG 1

#define DN_TRIALTYPE_NEW 0
#define DN_TRIALTYPE_RESUME 1

#define MAX_NAME_LEN 512



/*************************************************************
 *  Helper Functions
 *************************************************************/

/*
 * Get a list of all the input nodes to any given node
 */
idx_t getcontributors(su_mpi_model_l *m,
					  idx_t **sourcenodes, idx_t **numbufferspernode,
					  int commrank, int commsize)
{
	dnf_delaynet *dn = m->dn;

	/* Get the number of buffers per nodes globally */
	int numnodes = (int) dn->numnodes;

	if (commrank == 0) {
		int numnodes_g = 0;
		int *numnodes_l = malloc(sizeof(int)*commsize);
		int *nodeoffsets = calloc(commsize, sizeof(int));

		MPI_Gather(&numnodes,
				   1,
				   MPI_INT,
				   numnodes_l,
				   1,
				   MPI_INT,
				   0,
				   MPI_COMM_WORLD);

		for (idx_t i=0; i<commsize; i++)
			numnodes_g += numnodes_l[i];
		for (idx_t i=1; i<commsize; i++)
			nodeoffsets[i] = nodeoffsets[i-1] + numnodes_l[i-1];

		*numbufferspernode = malloc(sizeof(idx_t)*numnodes_g);
		MPI_Gatherv(dn->numbuffers,
				    numnodes,
					MPI_UNSIGNED_LONG, 
					*numbufferspernode,
					numnodes_l,
					nodeoffsets,
					MPI_UNSIGNED_LONG,
					0,
					MPI_COMM_WORLD);

		free(numnodes_l);
		free(nodeoffsets);
	} else {
		MPI_Gather(&numnodes,
				   1,
				   MPI_INT,
				   NULL,
				   1,
				   MPI_INT,
				   0,
				   MPI_COMM_WORLD);
		MPI_Gatherv(dn->numbuffers,
				    numnodes,
					MPI_UNSIGNED_LONG, 
					NULL,
					NULL,
					NULL,
					MPI_UNSIGNED_LONG,
					0,
					MPI_COMM_WORLD);
	}

	/* Get the buffer sources */
	idx_t numbuffers_g = 0;
	if (commrank == 0) {
		idx_t *numbufferstotal_perrank = malloc(sizeof(idx_t)*commsize);
		int *numbufferstotal_perrank_int = malloc(sizeof(int)*commsize);
		int *bufferoffsets_int = calloc(commsize, sizeof(int));

		MPI_Gather(&dn->numbufferstotal,
				   1,
				   MPI_UNSIGNED_LONG,
				   numbufferstotal_perrank,
				   1,
				   MPI_UNSIGNED_LONG,
				   0,
				   MPI_COMM_WORLD);

		for (idx_t i=0; i<commsize; i++) {
			numbufferstotal_perrank_int[i] = (int) numbufferstotal_perrank[i];
			numbuffers_g += numbufferstotal_perrank[i];
		}
		for (idx_t i=1; i<commsize; i++) {
			bufferoffsets_int[i] = bufferoffsets_int[i-1] +
								   numbufferstotal_perrank_int[i-1];
		}

		*sourcenodes = malloc(sizeof(idx_t)*numbuffers_g);
		MPI_Gatherv(dn->buffersourcenodes,
					(int) dn->numbufferstotal,
					MPI_UNSIGNED_LONG, 
					*sourcenodes,
					numbufferstotal_perrank_int,
					bufferoffsets_int,
					MPI_UNSIGNED_LONG,
					0,
					MPI_COMM_WORLD);
		free(numbufferstotal_perrank);
		free(numbufferstotal_perrank_int);
		free(bufferoffsets_int);
	} else {
		MPI_Gather(&dn->numbufferstotal,
				   1,
				   MPI_UNSIGNED_LONG,
				   NULL,
				   1,
				   MPI_UNSIGNED_LONG,
				   0,
				   MPI_COMM_WORLD);
		MPI_Gatherv(dn->buffersourcenodes,
				    (int) dn->numbufferstotal,
					MPI_UNSIGNED_LONG, 
					NULL,
					NULL,
					NULL,
					MPI_UNSIGNED_LONG,
					0,
					MPI_COMM_WORLD);
	}

	return numbuffers_g;
}



/*************************************************************
 *  Main
 *************************************************************/
int main(int argc, char *argv[])
{
	su_mpi_model_l *m;
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

	char *in_name = argv[1];
	printf("name: %s\n", in_name);

	/* model create or load */
	if (PG_MAIN_DEBUG) printf("Loading model on process %d\n", commrank);
	m = su_mpi_globalload(in_name, commrank, commsize);
	if (PG_MAIN_DEBUG) printf("Loaded model on process %d\n", commrank);

	/* set up spike recorder on each rank */
	char srname[MAX_NAME_LEN];
	char rankstr[MAX_NAME_LEN];
	sprintf(rankstr, "_%d_%d", commsize, commrank);
	strcpy(srname, in_name);
	strcat(srname, rankstr);
	strcat(srname, "_spikes.txt");
	spikerecord *sr = sr_init(srname, SPIKE_BLOCK_SIZE);

	/* Look of polychronous groups (PGs) */
	if (PG_MAIN_DEBUG) printf("Running simulation on rank %d\n", commrank);
	// su_mpi_runstdpmodel(m, tp, forced_inputs, numinputs,
	// 					sr, out_name, commrank, commsize, PROFILING);
	idx_t *sourcenodes = 0, *numbufferspernode = 0;
	idx_t numbuffers;
	numbuffers = getcontributors(m, &sourcenodes, &numbufferspernode, commrank, commsize);
	if (commrank == 0) printf("Number of buffers: %lu\n", numbuffers);
		


	/* Clean up */
	char srfinalname[256];
	strcpy(srfinalname, in_name);
	strcat(srfinalname, "_spikes.txt");

	if (PG_MAIN_DEBUG) printf("Saving spikes on rank %d\n", commrank);
	sr_collateandclose(sr, srfinalname, commrank, commsize, mpi_spike_type);
	if (PG_MAIN_DEBUG) printf("Saved spikes on rank %d\n", commrank);


	//if (PG_MAIN_DEBUG) printf("Freeing memory on rank 0 %d\n", commrank);
	//if (commrank == 0) {
	//	free(sourcenodes);
	//	free(numbufferspernode);
	//}
	//if (PG_MAIN_DEBUG) printf("Freed memory on rank 0 %d\n", commrank);

	if (PG_MAIN_DEBUG) printf("Freeing model on rank %d\n", commrank);
	su_mpi_freemodel_l(m);
	if (PG_MAIN_DEBUG) printf("Freed model on rank %d\n", commrank);

	MPI_Finalize();

	return 0;
}
