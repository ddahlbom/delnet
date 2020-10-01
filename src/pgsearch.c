#include "pgsearch.h"


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


idx_t *updateposition(idx_t *positions, idx_t numpositions,
				      idx_t which, idx_t maxidx)
{
	if (which == numpositions-1) {
		if (positions[which] == maxidx) {
			updateposition(positions, numpositions, which-1, maxidx);
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
			updateposition(positions, numpositions, which-1, maxidx);
			positions[which] = positions[which-1]+1;
		} else {
			positions[which] += 1;
		}
	}
	return positions;
}
