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
	printf("Rank: %d\n", commrank);

	/* Set up MPI Datatype for spikes */
	MPI_Datatype mpi_spike_type = sr_commitmpispiketype();

	/* Check number of inputs (needs one) */
	if (argc != 2) {
		printf("No model name given. Exiting...\n");
		exit(-1);
	}

	char *in_name = argv[1];
	printf("name: %s\n", in_name);

	/* Load model */
	if (PG_MAIN_DEBUG) printf("Loading model on process %d\n", commrank);
	m = su_mpi_globalload(in_name, commrank, commsize);
	if (PG_MAIN_DEBUG) printf("Loaded model on process %d\n", commrank);

	/* Set up trial parameters */
	su_mpi_trialparams tp;
	tp.dur = 0.07; // make parameter of executable
	tp.lambda = 0.01; 
	tp.randspikesize = 0.0;
	tp.randinput = 1.0;
	tp.inhibition = 1.0;
	tp.inputmode = 1.0;
	tp.multiinputmode = 4.0;
	tp.inputweight = 20.0;
	tp.recordstart = 0.0;
	tp.recordstop = 1000.0;
	tp.lambdainput = 1.0;
	tp.inputrefractorytime = 1.0;

	/* set up spike recorder on each rank */
	char srname[MAX_NAME_LEN];
	char rankstr[MAX_NAME_LEN];
	sprintf(rankstr, "_%d_%d", commsize, commrank);
	strcpy(srname, in_name);
	strcat(srname, rankstr);
	strcat(srname, "_spikes.txt");
	spikerecord *sr = sr_init(srname, SPIKE_BLOCK_SIZE);

	/* Make list of contributing neurons */
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
	idx_t maxgroups = 10000;
	idx_t numgroups = 0; // running tally of groups found
	su_mpi_input input_l;
	su_mpi_spike *inputspikes = malloc(sizeof(su_mpi_spike)*groupsize);
	data_t maxdelay = 0;

	if (commrank == 0) {
		idx_t *positions = malloc(sizeof(idx_t)*groupsize);
		idx_t *positions_old = malloc(sizeof(idx_t)*groupsize);

		int done = 0;
		int dotrial;
		int globaldone = 0;
		idx_t new = 0;
		idx_t offset = 0;
		idx_t maxidx = 0;
		data_t threshold = 19.0;
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
						/* here run partial simulation with group as input */
						free(inputspikes);
						inputspikes = malloc(sizeof(su_mpi_spike)*groupsize);
						maxdelay = 0.0;
						for (idx_t n=0; n<groupsize; n++) {
							inputspikes[n].i = sourcenodes[offset+positions[n]];
							inputspikes[n].t = ((data_t) delays[offset+positions[n]])/m->p.fs;
							if (inputspikes[n].t > maxdelay) maxdelay = inputspikes[n].t;
						}
						for (idx_t n=0; n<groupsize; n++)
							inputspikes[n].t = maxdelay - inputspikes[n].t;
						dotrial=1;
						MPI_Bcast(&dotrial, 1, MPI_INT, 0, MPI_COMM_WORLD);
						MPI_Bcast(inputspikes, groupsize, mpi_spike_type,
								  0, MPI_COMM_WORLD);
						input_l.len = pruneinputtolocal(&inputspikes, groupsize, m);
						input_l.spikes = inputspikes;

						MPI_Barrier(MPI_COMM_WORLD);
						for (idx_t z=0; z<groupsize; z++) {
							printf("%lu with weight %g\n",
								   sourcenodes[offset+positions[z]],
								   weights[offset+positions[z]]);
						}
						for (idx_t z=0; z<input_l.len; z++) {
							printf("%lu @ %g (rank %d)\n", input_l.spikes[z].i,
												 input_l.spikes[z].t, commrank);
						}
						MPI_Barrier(MPI_COMM_WORLD);
						
						/* Reset buffers and neuron states between trials */
						for(idx_t z=0; z<m->maxnode; z++) {
							m->neurons[z].v = m->neurons[z].c;
							m->neurons[z].u = m->neurons[z].b * m->neurons[z].c;
						}
						for(idx_t z=0; z<m->dn->numbufferstotal; z++)
							dnf_bufinit(&m->dn->buffers[z], m->dn->buffers[z].delaylen);

						printf("Running trial %lu (%d)\n", numgroups, commrank);
						su_mpi_runpgtrial(m, tp, &input_l, 1, sr, in_name,
										  numgroups*tp.dur, commrank, commsize); 
						printf("Ran trial %lu (%d)\n", numgroups, commrank);

						numgroups += 1; // change so only if group accepted
						if (numgroups == maxgroups) {
							printf("Hit max number of groups!\n");
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
		dotrial=0;
		MPI_Bcast(&dotrial, 1, MPI_INT, 0, MPI_COMM_WORLD);
		printf("number of anchor groups: %lu\n", numgroups);
		free(positions);
		free(positions_old);
	} else {
		idx_t numgroups = 0;
		int notfinished; // 1 = run trial, 0 = stop loop
		while (1) {
			MPI_Bcast(&notfinished, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (notfinished) {
				free(inputspikes);
				inputspikes = malloc(sizeof(su_mpi_spike)*groupsize);
				MPI_Bcast(inputspikes, groupsize, mpi_spike_type,
						  0, MPI_COMM_WORLD);
				input_l.len = pruneinputtolocal(&inputspikes, groupsize, m);
				input_l.spikes = inputspikes;

				MPI_Barrier(MPI_COMM_WORLD);
				for (idx_t z=0; z <input_l.len; z++) {
					printf("%lu @ %g (rank %d)\n", input_l.spikes[z].i,
										 		   input_l.spikes[z].t, commrank);
				}
				MPI_Barrier(MPI_COMM_WORLD);
				
				/* Reset buffers and neuron states between trials */
				for(idx_t z=0; z<m->maxnode; z++) {
					m->neurons[z].v = m->neurons[z].c;
					m->neurons[z].u = m->neurons[z].b * m->neurons[z].c;
				}
				for(idx_t z=0; z<m->dn->numbufferstotal; z++)
					dnf_bufinit(&m->dn->buffers[z], m->dn->buffers[z].delaylen);
				su_mpi_runpgtrial(m, tp, &input_l, 1, sr, in_name,
								  numgroups*tp.dur, commrank, commsize); 
				numgroups += 1;
			} else 
				break;
		}
	}



	/* Clean up */

	printf("Freeing spike records (%d)\n", commrank);
	char srfinalname[256];
	strcpy(srfinalname, in_name);
	strcat(srfinalname, "_spikes.txt");
	sr_collateandclose(sr, srfinalname, commrank, commsize, mpi_spike_type);
	printf("Freed spike records (%d)\n", commrank);

	if (commrank == 0) {
		//printf("Freeing misc allocations (%d)\n", commrank);
		free(sourcenodes);
		free(numbufferspernode);
		free(weights);
		free(delays);
		//printf("Freed misc allocations (%d)\n", commrank);
	}

	//printf("Freeing inputspikes (%d)\n", commrank);
	free(inputspikes);
	//printf("Freed inputspikes (%d)\n", commrank);

	//printf("Freeing the model (%d)\n", commrank);
	su_mpi_freemodel_l(m);
	//printf("Freed the model (%d)\n", commrank);

	//printf("MPI_Finalize() (%d)\n", commrank);
	MPI_Finalize();
	//printf("MPI_Finalized() (%d)\n", commrank);

	return 0;
}
