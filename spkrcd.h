#ifndef SPKRCD_H
#define SPKRCD_H

#define SPIKE_BLOCK_SIZE 8192
#define SR_FLOAT_T float


/*
 * -------------------- Structures --------------------
 */
typedef struct spike_s {
	int neuron;
	SR_FLOAT_T time;
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
void sr_save_spike(spikerecord *sr, int neuron, SR_FLOAT_T time);
void sr_close(spikerecord *sr);
void sr_collateandclose(spikerecord *sr, char *finalfilename, int commrank, int commsize);


#endif
