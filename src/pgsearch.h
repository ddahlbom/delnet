#ifndef PGSEARCH_H
#define PGSEARCH_H

#include <stdlib.h>
#include <mpi.h>

#include "delnet.h"
#include "simutils.h"


/*************************************************************
 *  Function Declarations
 *************************************************************/
idx_t getcontributors(su_mpi_model_l *m,
					  idx_t **sourcenodes, idx_t **numbufferspernode,
					  int commrank, int commsize);

idx_t *updateposition(idx_t *positions, idx_t numpositions,
				    idx_t which, idx_t maxidx);

#endif
