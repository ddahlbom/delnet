#ifndef PGSEARCH_H
#define PGSEARCH_H

#include <stdlib.h>
#include <mpi.h>

#include "delnet.h"
#include "simutils.h"
#include "inputs.h"

#define PG_SEARCH_HIT_LIMIT 0
#define PG_SEARCH_FINISHED 1


/*************************************************************
 *  Function Declarations
 *************************************************************/
idx_t pg_getcontributors(su_mpi_model_l *m,
	    			     idx_t **sourcenodes,
	    			     idx_t **numbufferspernode,
	    			     int commrank, int commsize);

idx_t pg_getcontributorweights(su_mpi_model_l *m,  data_t **weights, int commrank, int commsize);

idx_t pg_getcontributordelays(su_mpi_model_l *m,  delay_t **delays, int commrank, int commsize);

idx_t *pg_updateposition(idx_t *positions, idx_t numpositions,
				    idx_t which, idx_t maxidx);

int pg_findpgs(su_mpi_model_l *m, su_mpi_trialparams tp, idx_t groupsize, data_t threshold, idx_t maxgroups,
				spikerecord *sr, MPI_Datatype mpi_spike_type, int commrank, int commsize);

#endif
