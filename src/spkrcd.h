#ifndef SPKRCD_H
#define SPKRCD_H


#define SPIKE_BLOCK_SIZE 8192


/*
 * -------------------- Structures --------------------
 */
typedef struct spike_s {
	unsigned long neuron;
	double time;
} spike;

typedef struct spikerecord_s {
	spike *spikes;	
	char *filename;
	char *writemode;
	size_t blockcount;
	size_t numspikes;
	size_t blocksize;
} spikerecord;


/*
 * -------------------- Function Declarations --------------------
 */


spikerecord *sr_init(char *filename, size_t spikes_in_block);
void sr_save_spike(spikerecord *sr, unsigned long neuron, double time);
void sr_close(spikerecord *sr);
void sr_collateandclose(spikerecord *sr, char *finalfilename, int commrank, int commsize, MPI_Datatype mpi_spike_type);
int *len_to_offsets(int *lens, int n);
MPI_Datatype sr_commitmpispiketype();


#endif
