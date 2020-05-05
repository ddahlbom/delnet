#include <stdlib.h>
#include <stdio.h>

#include "spkrcd.h"

spikerecord *sr_init(char *filename, size_t spikes_in_block)
{
	spikerecord *sr;

	sr = malloc(sizeof(spikerecord));
	sr->blocksize = spikes_in_block;
	sr->numspikes = 0;
	sr->blockcount = 0;
	sr->filename = filename;
	sr->writemode = "w";
	sr->spikes = malloc(sizeof(spike)*spikes_in_block);

	return sr;
}

void sr_save_spike(spikerecord *sr, int neuron, SR_FLOAT_T time)
{
	if (sr->blockcount < sr->blocksize) {
		sr->spikes[sr->blockcount].neuron = neuron;
		sr->spikes[sr->blockcount].time = time;
		sr->blockcount += 1;
		sr->numspikes += 1;
	} 
	else {
		FILE *spike_file;
		spike_file = fopen(sr->filename, sr->writemode);
		for (int i=0; i < sr->blocksize; i++) {
			fprintf(spike_file, "%f  %d\n", 
					sr->spikes[i].time,
					sr->spikes[i].neuron);	
		}
		fclose(spike_file);

		sr->writemode = "a";
		sr->spikes[0].neuron = neuron;
		sr->spikes[0].time = time;
		sr->blockcount = 1;
		sr->numspikes += 1;
	}
}


void sr_close(spikerecord *sr)
{
	/* write any as-of-yet unsaved spikes */	
	FILE *spike_file;
	spike_file = fopen(sr->filename, sr->writemode);
	for (int i=0; i < sr-> numspikes % sr->blocksize; i++) {
		fprintf(spike_file, "%f  %d\n", 
				sr->spikes[i].time,
				sr->spikes[i].neuron);	
	}
	fclose(spike_file);

	/* free memory */
	free(sr->spikes);
	free(sr);
}
