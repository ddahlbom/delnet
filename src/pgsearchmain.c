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
#include "inputs.h"

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

	/* Consolidate delay times */
	delay_t *delays = 0;
	getcontributordelays(m, &delays, commrank, commsize);

	/* Iterate through combinations and test them */
	idx_t groupsize = 3;
	idx_t maxgroups = 100000;
	idx_t numgroups = 0;
	//idx_t numinputs_l = 0;

	su_mpi_input input_g, input_l;
	input_g.len = groupsize;
	su_mpi_spike *inputspikes = malloc(sizeof(su_mpi_spike)*groupsize);
	data_t *spiketimes = malloc(sizeof(data_t)*groupsize);
	data_t maxdelay = 0;

	su_mpi_trialparams tp;
	tp.dur = 0.07; // make variable
	tp.lambda = 0.01; 
	tp.randspikesize = 0.0;
	tp.randinput = 0.0;
	tp.inhibition = 1.0;
	tp.inputmode = 2.0;
	tp.multiinputmode = 1.0;
	tp.inputweight = 20.0;
	tp.recordstart = 0.0;
	tp.recordstop = tp.dur;
	tp.lambdainput = 1.0;
	tp.inputrefractorytime = 1.0;

	if (commrank == 0) {
		threegroup *pgs = malloc(sizeof(threegroup)*maxgroups);
		idx_t *positions = malloc(sizeof(idx_t)*groupsize);
		idx_t *positions_old = malloc(sizeof(idx_t)*groupsize);

		int done = 0;
		int dotrial = 1;
		int globaldone = 0;
		MPI_Request donereq, trialreq;
		idx_t new = 0;
		idx_t offset = 0;
		idx_t maxidx = 0;
		data_t threshold = 19.0;
		data_t totalweight = 0.0;

		/* Iterate through nodes, test combinations of their contributors */
		printf("TEST: Starting search...\n");
		for (idx_t i=0; i<numnodes_g; i++) {
			printf("Check inputs to neuron %lu\n", i);
			if (globaldone) {
				break;
			}
			offset = bufstartidcs[i];	
			maxidx = numbufferspernode[i]-1;
			for (idx_t j=0; j<groupsize; j++) {
				positions[j] = j;
				positions_old[j] = 0;
			}
			done = 0;
			while (!done) {
				/* Check if new position. Repeat only when combos exhausted. */
				new = 0;
				for (idx_t k=0; k<groupsize; k++)
					new += positions_old[k] == positions[k];
				if (new == groupsize) {
					done = 1;
					break;
				} else {
					/* Test weights and run trial if necessary */
					totalweight = 0.0;
					for (idx_t l=0; l<groupsize; l++) 
						totalweight += weights[offset+positions[l]];
					if (totalweight > threshold) {
						printf("Found a combo -- testing\n");
						for (idx_t r=1; r<commsize; r++) {
							MPI_Isend(&dotrial, 1, MPI_INT, r, 1,
									  MPI_COMM_WORLD, &trialreq);
						}
						/* here run partial simulation with group as input */
						free(inputspikes);
						inputspikes = malloc(sizeof(su_mpi_spike)*groupsize);
						maxdelay = 0;
						for (idx_t n=0; n<groupsize; n++) {
							spiketimes[n] = (data_t) delays[offset+positions[n]]
											/ m->p.fs;
							if (spiketimes[n] > maxdelay)
								maxdelay = spiketimes[n];
						}
						for (idx_t n=0; n<groupsize; n++)
							inputspikes[n].t = maxdelay - inputspikes[n].t;
						MPI_Bcast(inputspikes, groupsize, mpi_spike_type,
								  0, MPI_COMM_WORLD);
						input_l.len = pruneinputtolocal(&inputspikes,
														groupsize, m);
						input_l.spikes = inputspikes;
						
						// Run the trial...
						su_mpi_runpgtrial(m,tp, &input_l, 1, sr, "test",
										  commrank, commsize); 


						numgroups += 1; // change so only if group accepted
						if (numgroups == maxgroups) {
							printf("Hit max number of groups!");
							done = 1;
							globaldone = 1;
							i = numnodes_g;
						}
					}
				}

				/* Find a new combination */
				for (idx_t m=0; m<groupsize; m++)
					positions_old[m] = positions[m];	
				updateposition(positions, groupsize, groupsize-1, maxidx); 

			}
		}
		for (idx_t r=1; r<commsize; r++) {
			MPI_Isend(&done, 1, MPI_INT, r, 0,
					  MPI_COMM_WORLD, &donereq);
		}
		printf("number of anchor groups: %lu\n", numgroups);
		free(pgs);
		free(positions);
		free(positions_old);
	} else {
		MPI_Request trialreq;
		MPI_Request donereq;
		int done = false;
		int dotrial = false;
		int trialflag=0, doneflag=0;
		MPI_Irecv(&done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &donereq);
		MPI_Irecv(&dotrial, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &trialreq);
		while (1) {
			MPI_Test(&trialreq, &trialflag, MPI_STATUS_IGNORE);
			if (trialflag) {
				MPI_Bcast(inputspikes, groupsize, mpi_spike_type,
						  0, MPI_COMM_WORLD);
				dotrial=false;
				input_l.len = pruneinputtolocal(&inputspikes,
												groupsize, m);
				input_l.spikes = inputspikes;
				
				// Run the trial...
				su_mpi_runpgtrial(m,tp, &input_l, 1, sr, "test",
								  commrank, commsize); 
				MPI_Irecv(&dotrial, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &trialreq);
			}
			MPI_Test(&donereq, &doneflag, MPI_STATUS_IGNORE);
			if (doneflag) {
				break;
			}
		}
	}


	/* Clean up */
	char srfinalname[256];
	strcpy(srfinalname, in_name);
	strcat(srfinalname, "_spikes.txt");

	sr_collateandclose(sr, srfinalname, commrank, commsize, mpi_spike_type);
	if (commrank == 0) {
		free(sourcenodes);
		free(numbufferspernode);
		free(weights);
		free(delays);
	}
	su_mpi_freemodel_l(m);
	MPI_Finalize();

	return 0;
}
