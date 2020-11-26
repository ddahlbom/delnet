#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <mpi.h>

#include "delnet.h"
#include "spkrcd.h"
#include "pgsearch.h"

#define MAX_NAME_LEN 512

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
    if (argc != 6) {
        printf("Need five arguments: (1) model name; (2) base size; (3) threshold; (4) duration; (5) max number of groups.\n");
        exit(-1);
    }

    char *in_name = argv[1];

    /* Load model */
    m = su_mpi_globalload(in_name, commrank, commsize);

    /* Set up trial parameters */
    su_mpi_trialparams tp;
    tp.dur = atof(argv[4]); // make parameter of executable
    tp.lambda = 0.01; 
    tp.randspikesize = 0.0;
    tp.randinput = 1.0;
    tp.inhibition = 1.0;
    tp.inputmode = 1.0;
    tp.multiinputmode = 4.0; // one-shot input at start
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

    /* perform the search */
    pg_findpgs(m, tp, atoi(argv[2]), atof(argv[3]), atoi(argv[5]), sr, mpi_spike_type, commrank, commsize);

    /* Clean up */
    char srfinalname[256];
    strcpy(srfinalname, in_name);
    strcat(srfinalname, "pg_spikes.txt");
    sr_collateandclose(sr, srfinalname, commrank, commsize, mpi_spike_type);

    su_mpi_freemodel_l(m);

    MPI_Finalize();

    return 0;
}
