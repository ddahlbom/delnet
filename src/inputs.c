#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "inputs.h"

long int pruneinputtolocal(su_mpi_spike **forced_input, idx_t inputlen, su_mpi_model_l *m) {
    idx_t nlocal=0;
    idx_t i1, i2;
    i1 = m->dn->nodeoffsetglobal;
    i2 = i1 + m->dn->numnodes;

    for (idx_t n=0; n<inputlen; n++) 
        if (i1 <= (*forced_input)[n].i && (*forced_input)[n].i < i2) nlocal += 1;

    su_mpi_spike *input_local = 0;
    idx_t c = 0;
    input_local = malloc(sizeof(su_mpi_spike)*nlocal);
    for (idx_t n=0; n<inputlen; n++) {
        if (i1 <= (*forced_input)[n].i && (*forced_input)[n].i < i2) {
            input_local[c].i = (*forced_input)[n].i-i1; // ... - i1: put into local indexing basis
            input_local[c].t = (*forced_input)[n].t;
            c++;
        }
    }
    free(*forced_input);
    *forced_input = input_local;
    
    return nlocal;
}


long int loadinputs(char *in_name, su_mpi_input **forced_input, MPI_Datatype mpi_spike_type, su_mpi_model_l *m, int commrank)
{
    FILE *infile;
    long int inputlen;
    long int prunedlen;
    long int numinputs;
    size_t loadsize;
    if (commrank == 0) {
        /* Read the input file data */
        char infilename[MAX_NAME_LEN];
        strcpy(infilename, in_name);
        strcat(infilename, "_input.bin");
        infile = fopen(infilename, "rb");
        checkfileload(infile, infilename);

        /* Set up array of input structures */
        loadsize = fread(&numinputs, sizeof(long int), 1, infile);
        if (loadsize != 1) {printf("Failed to load number of inputs\n"); exit(-1); }
        MPI_Bcast(&numinputs, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        *forced_input = malloc(sizeof(su_mpi_input)*numinputs);

        /* Load each input */
        for (idx_t n=0; n<numinputs; n++) {
            /* find out number of spikes in input and allocate */
            loadsize = fread(&inputlen, sizeof(long int), 1, infile);
            if (loadsize != 1) {printf("Failed to load input\n"); exit(-1); }
            (*forced_input)[n].len = inputlen; 
            (*forced_input)[n].spikes = malloc(sizeof(su_mpi_spike)*inputlen);

            /* broadcast input */
            loadsize = fread((*forced_input)[n].spikes, sizeof(su_mpi_spike), inputlen, infile);
            if (loadsize != inputlen) {printf("Failed to load input\n"); exit(-1); }

            /* Broadcast to all ranks */
            MPI_Bcast( &inputlen, 1, MPI_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast( (*forced_input)[n].spikes, inputlen, mpi_spike_type, 0, MPI_COMM_WORLD);
            prunedlen = pruneinputtolocal(&(*forced_input)[n].spikes, (*forced_input)[n].len, m);
            (*forced_input)[n].len = prunedlen;
        }
        fclose(infile);
    } else {
        /* Get number of inputs and allocate */
        MPI_Bcast(&numinputs, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        *forced_input = malloc(sizeof(su_mpi_input)*numinputs);

        for (idx_t n=0; n<numinputs; n++) {
            /* Recieve input from rank 0 */
            MPI_Bcast( &inputlen, 1, MPI_LONG, 0, MPI_COMM_WORLD);
            (*forced_input)[n].len = inputlen;
            (*forced_input)[n].spikes = malloc(sizeof(su_mpi_spike)*inputlen);
            MPI_Bcast( (*forced_input)[n].spikes, inputlen, mpi_spike_type, 0, MPI_COMM_WORLD);
            prunedlen = pruneinputtolocal(&(*forced_input)[n].spikes, (*forced_input)[n].len, m);
            (*forced_input)[n].len = prunedlen;
        }
    }

    return numinputs;
}

void freeinputs(su_mpi_input **forced_input, idx_t numinputs) {
    for (idx_t n=0; n<numinputs; n++) 
        free((*forced_input)[n].spikes);
    free(*forced_input);
}
