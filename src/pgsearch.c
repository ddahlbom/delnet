#include "pgsearch.h"


/*
 * Get a list of all the input nodes to any given node
 */
idx_t pg_getcontributors(su_mpi_model_l *m,
					  	 idx_t **sourcenodes,
					  	 idx_t **numbufferspernode,
					  	 int commrank, int commsize)
{
	dnf_delaynet *dn = m->dn;

	/* Get the number of buffers per nodes globally */
	int numnodes = (int) dn->numnodes;
	idx_t numnodes_g = 0;

	if (commrank == 0) {
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
			numnodes_g += (idx_t) numnodes_l[i];
		for (idx_t i=1; i<commsize; i++)
			nodeoffsets[i] = nodeoffsets[i-1] + numnodes_l[i-1];
		MPI_Bcast(&numnodes_g, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

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
		MPI_Bcast(&numnodes_g, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
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

	return numnodes_g;
}


/*
 * Get a list of all the input nodes to any given node
 */
idx_t pg_getcontributorweights(su_mpi_model_l *m,  data_t **weights, int commrank, int commsize)
{
	dnf_delaynet *dn = m->dn;

	/* Get the synapse weights */
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

		*weights = malloc(sizeof(idx_t)*numbuffers_g);
		MPI_Gatherv(m->synapses,
					(int) dn->numbufferstotal,
					MPI_DOUBLE, 
					*weights,
					numbufferstotal_perrank_int,
					bufferoffsets_int,
					MPI_DOUBLE,
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
		MPI_Gatherv(m->synapses,
				    (int) dn->numbufferstotal,
					MPI_DOUBLE, 
					NULL,
					NULL,
					NULL,
					MPI_DOUBLE,
					0,
					MPI_COMM_WORLD);
	}
	return numbuffers_g;
}



/*
 * Get a list of all the delay times 
 */
idx_t pg_getcontributordelays(su_mpi_model_l *m,  delay_t **delays, int commrank, int commsize)
{
	dnf_delaynet *dn = m->dn;
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

		delay_t *delays_l = malloc(sizeof(delay_t)*dn->numbufferstotal);
		for (idx_t i=0; i<dn->numbufferstotal; i++)
			delays_l[i] = dn->buffers[i].delaylen;

		*delays = malloc(sizeof(delay_t)*numbuffers_g);
		MPI_Gatherv(delays_l,
					(int) dn->numbufferstotal,
					MPI_UNSIGNED_SHORT, 
					*delays,
					numbufferstotal_perrank_int,
					bufferoffsets_int,
					MPI_UNSIGNED_SHORT,
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
		delay_t *delays_l = malloc(sizeof(delay_t)*dn->numbufferstotal);
		for (idx_t i=0; i<dn->numbufferstotal; i++)
			delays_l[i] = dn->buffers[i].delaylen;
		MPI_Gatherv(delays_l,
				    (int) dn->numbufferstotal,
					MPI_UNSIGNED_SHORT, 
					NULL,
					NULL,
					NULL,
					MPI_UNSIGNED_SHORT,
					0,
					MPI_COMM_WORLD);
		free(delays_l);
	}
	return numbuffers_g;
}



/* Advance one step through all combinations */
idx_t *pg_updateposition(idx_t *positions, idx_t numpositions,
				         idx_t which, idx_t maxidx)
{
	if (which == numpositions-1) {
		if (positions[which] == maxidx) {
			pg_updateposition(positions, numpositions, which-1, maxidx);
			positions[which] = positions[which-1] + 1;
		} else {
			positions[which] += 1;
		}
	} else if (which == 0) {
		if (positions[which] == positions[which+1]-1) {
			return positions;
		} else {
			positions[which] += 1;
		}
	} else {
		if (positions[which] == positions[which+1]-1) {
			pg_updateposition(positions, numpositions, which-1, maxidx);
			positions[which] = positions[which-1]+1;
		} else {
			positions[which] += 1;
		}
	}
	return positions;
}



int pg_findpgs(su_mpi_model_l *m, su_mpi_trialparams tp, spikerecord *sr,
			   MPI_Datatype mpi_spike_type, int commrank, int commsize)
{
	/* Make list of contributing neurons */
	idx_t *sourcenodes = 0, *numbufferspernode = 0;
	idx_t numnodes_g, numbuffers_g=0;
	numnodes_g = pg_getcontributors(m, &sourcenodes, &numbufferspernode, commrank, commsize);


	/* Consolidate synapse weights */
	data_t *weights = 0;
	numbuffers_g = pg_getcontributorweights(m, &weights, commrank, commsize);

	/* Consolidate delay times */
	delay_t *delays = 0;
	pg_getcontributordelays(m, &delays, commrank, commsize);

	/* Iterate through combinations and test them */
	idx_t groupsize = 3;
	idx_t maxgroups = 100;
	idx_t numgroups = 0; // running tally of groups found
	su_mpi_input input_l;
	su_mpi_spike *inputspikes = 0;
	data_t maxdelay = 0;
	idx_t *bufstartidcs = 0;

	if (commrank == 0) {
		int done = 0;
		int dotrial;
		int globaldone = 0;
		idx_t new = 0;
		idx_t offset = 0;
		idx_t maxidx = 0;
		data_t threshold = 19.0;
		data_t totalweight = 0.0;

		idx_t *positions = malloc(sizeof(idx_t)*groupsize);
		idx_t *positions_old = malloc(sizeof(idx_t)*groupsize);

		/* Get buffer offset idices */
		bufstartidcs = malloc(numnodes_g*sizeof(idx_t));
		bufstartidcs[0] = 0;

		numbuffers_g += numbufferspernode[0];
		for (idx_t i=1; i<numnodes_g; i++) {
			bufstartidcs[i] = bufstartidcs[i-1] + numbufferspernode[i-1];
			numbuffers_g += numbufferspernode[i];
		}

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

						/* Reset buffers and neuron states between trials */
						for(idx_t z=0; z<m->maxnode; z++) {
							m->neurons[z].v = m->neurons[z].c;
							m->neurons[z].u = m->neurons[z].b * m->neurons[z].c;
						}
						for(idx_t z=0; z<m->dn->numbufferstotal; z++)
							dnf_bufinit(&m->dn->buffers[z], m->dn->buffers[z].delaylen);

						printf("Running trial %lu (%d)\n", numgroups, commrank);
						su_mpi_runpgtrial(m, tp, &input_l, 1, sr,
										  numgroups*tp.dur, commrank, commsize); 
						printf("Ran trial %lu (%d)\n", numgroups, commrank);

						numgroups += 1; // change so only if group accepted
						free(inputspikes);
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
				pg_updateposition(positions, groupsize, groupsize-1, maxidx); 
			}
		}
		dotrial=0;
		MPI_Bcast(&dotrial, 1, MPI_INT, 0, MPI_COMM_WORLD);
		printf("number of anchor groups: %lu\n", numgroups);
		free(positions);
		free(positions_old);
		free(sourcenodes);
		free(numbufferspernode);
		free(weights);
		free(delays);
		free(bufstartidcs);
	} else {
		idx_t numgroups = 0;
		int notfinished; // 1 = run trial, 0 = stop loop
		while (1) {
			MPI_Bcast(&notfinished, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (notfinished) {
				inputspikes = malloc(sizeof(su_mpi_spike)*groupsize);
				MPI_Bcast(inputspikes, groupsize, mpi_spike_type,
						  0, MPI_COMM_WORLD);
				input_l.len = pruneinputtolocal(&inputspikes, groupsize, m);
				input_l.spikes = inputspikes;

				/* Reset buffers and neuron states between trials */
				for(idx_t z=0; z<m->maxnode; z++) {
					m->neurons[z].v = m->neurons[z].c;
					m->neurons[z].u = m->neurons[z].b * m->neurons[z].c;
				}
				for(idx_t z=0; z<m->dn->numbufferstotal; z++)
					dnf_bufinit(&m->dn->buffers[z], m->dn->buffers[z].delaylen);
				su_mpi_runpgtrial(m, tp, &input_l, 1, sr,
								  numgroups*tp.dur, commrank, commsize); 
				numgroups += 1;
				free(inputspikes);
			} else 
				break;
		}
	}
	return 0;
}
