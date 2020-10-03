#ifndef INPUTS_H
#define INPUTS_H
#include "simutils.h"


long int pruneinputtolocal(su_mpi_spike **forced_input, idx_t inputlen, su_mpi_model_l *m) ;

long int loadinputs(char *in_name, su_mpi_input **forced_input, MPI_Datatype mpi_spike_type, su_mpi_model_l *m, int commrank);

void freeinputs(su_mpi_input **forced_input, idx_t numinputs);


#endif
