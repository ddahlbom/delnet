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
#include "pgsearch.h"

#define PROFILING 1
#define PG_MAIN_DEBUG 1

#define DN_TRIALTYPE_NEW 0
#define DN_TRIALTYPE_RESUME 1

#define MAX_NAME_LEN 512


typedef struct threegroup_s {
	idx_t members[3];
} threegroup;


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

	/* Get incoming neurons */
	if (PG_MAIN_DEBUG) printf("Starting PG search process... %d\n", commrank);
	idx_t *sourcenodes = 0, *numbufferspernode = 0;
	idx_t numnodes_g, numbuffers_g=0;
	numnodes_g = getcontributors(m, &sourcenodes, &numbufferspernode, commrank, commsize);
	idx_t *bufstartidcs = malloc(numnodes_g*sizeof(idx_t));
	bufstartidcs[0] = 0;

	if (commrank == 0) {
		numbuffers_g += numbufferspernode[0];
		for (idx_t i=1; i<numnodes_g; i++) {
			bufstartidcs[i] = bufstartidcs[i-1] + numbufferspernode[i-1];
			numbuffers_g += numbufferspernode[i];
		}
		printf("TEST: Total # of synapses: %lu\n", numbuffers_g);
	}

	/* Consolidate synapse weights */
	data_t *weights = 0;
	numbuffers_g = getcontributorweights(m, &weights, commrank, commsize);

	/* Iterate through combinations and test them */
	idx_t groupsize = 3;
	idx_t maxgroups = 10000;
	idx_t numgroups = 0;

	if (commrank == 0) {
		threegroup *pgs = malloc(sizeof(threegroup)*maxgroups);
		idx_t *positions = malloc(sizeof(idx_t)*groupsize);
		idx_t *positions_old = malloc(sizeof(idx_t)*groupsize);

		bool done = false;
		idx_t new = 0;
		idx_t offset = 0;
		idx_t maxidx = 0;
		data_t threshold = 18.8;
		data_t totalweight = 0.0;

		/* Iterate through nodes, test combinations of their contributors */
		printf("TEST: Starting search...\n");
		for (idx_t i=0; i<numnodes_g; i++) {
			offset = bufstartidcs[i];	
			maxidx = numbufferspernode[i]-1;
			for (idx_t j=0; j<groupsize; j++) {
				positions[j] = j;
				positions_old[j] = 0;
			}
			while (!done) {
				/* Make sure its new -- only repeats when combos exhausted */
				new = 0;
				for (idx_t k=0; k<groupsize; k++)
					new += positions_old[k] == positions[k];
				if (new == groupsize) {
					done = true;
				} else {
					/* Test weights and run trial if necessary */
					totalweight = 0.0;
					for (idx_t l=0; l<groupsize; l++) 
						totalweight += weights[offset+positions[l]];
					if (totalweight > threshold) {
						/* here run partial simulation with group as input */
						/* going to need to recover delays as well... */
						numgroups += 1;
						printf("Candidate group: %lu %lu %lu\n", 
								sourcenodes[offset+positions[0]],
								sourcenodes[offset+positions[1]],
								sourcenodes[offset+positions[2]]);
					}
				}
				/* Find a new combination */
				for (idx_t m=0; m<groupsize; m++)
					positions_old[m] = positions[m];	
				updateposition(positions, groupsize, groupsize-1, maxidx); 
			}
		}
		printf("number of anchor groups: %lu\n", numgroups);
	} else {
	}


	/* Clean up */
	char srfinalname[256];
	strcpy(srfinalname, in_name);
	strcat(srfinalname, "_spikes.txt");

	sr_collateandclose(sr, srfinalname, commrank, commsize, mpi_spike_type);
	if (commrank == 0) {
		free(sourcenodes);
		free(numbufferspernode);
	}
	su_mpi_freemodel_l(m);
	MPI_Finalize();

	return 0;
}
