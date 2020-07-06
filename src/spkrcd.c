#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <mpi.h>

#include "spkrcd.h"


#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "what is happening here?"
#endif

spikerecord *sr_init(char *filename, size_t spikes_in_block)
{
	spikerecord *sr;

	sr = malloc(sizeof(spikerecord));
	sr->blocksize = spikes_in_block;
	sr->numspikes = 0;
	sr->blockcount = 0;
	sr->filename = filename;
	sr->writemode = "w"; 	// so initial write destroys previous
	sr->spikes = malloc(sizeof(spike)*spikes_in_block);

	return sr;
}

void sr_save_spike(spikerecord *sr, unsigned long neuron, double time)
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
			fprintf(spike_file, "%f  %lu\n", 
					sr->spikes[i].time,
					sr->spikes[i].neuron);	
		}
		fclose(spike_file);

		sr->writemode = "a"; 	// always append after first write
		sr->spikes[0].neuron = neuron;
		sr->spikes[0].time = time;
		sr->blockcount = 1;
		sr->numspikes += 1;
	}
}

int *len_to_offsets(int *lens, int n)
{
	int *offsets = calloc(n, sizeof(int));

	for (int i=1; i<n; i++) {
		for (int j=1; j<=i; j++) {
			offsets[i] += lens[j-1];
		}
	}
	return offsets;
}


static spike *scmp1 = 0;
static spike *scmp2 = 0;
static double diff = 0.0;

int spkcomp(const void *spike1, const void * spike2) {
	scmp1 = (spike *)spike1;
	scmp2 = (spike *)spike2;
	diff = scmp1->time - scmp2->time;

	if (diff < 0)
		return -1;
	else if (diff > 0)
		return 1;
	else 
		return scmp1->neuron - scmp2->neuron;
}

MPI_Datatype sr_commitmpispiketype()
{
	const int 		nitems = 2;
	int 			blocklengths[2] = {1, 1};
	MPI_Datatype	types[2] = {MPI_UNSIGNED_LONG, MPI_DOUBLE};
	MPI_Datatype	mpi_spike_type;
	MPI_Aint  		offsets[2];

	offsets[0] = offsetof(spike, neuron);
	offsets[1] = offsetof(spike, time);
	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_spike_type);
	if (MPI_SUCCESS != MPI_Type_commit(&mpi_spike_type)) {
		printf("Failed to commit custom MPI spike type!\n");
		exit(-1);
	}

	return mpi_spike_type;
}

/*
 * Read spikes written by each process into memory, sort by time,
 * write into a unified file, delete individual files. 
 *
 * Very naive, essentially sequential approach, refine later.
 */
void sr_collateandclose(spikerecord *sr, char *finalfilename,
						int commrank, int commsize,
						MPI_Datatype mpi_spike_type)
{
	/* Set up MPI Datatype for spikes */
	/* Each rank finishing writing its local file */		
	FILE *spike_file;
	spike_file = fopen(sr->filename, sr->writemode);
	for (int i=0; i < sr-> numspikes % sr->blocksize; i++) {
		fprintf(spike_file, "%lf  %lu\n", 
				sr->spikes[i].time,
				sr->spikes[i].neuron);	
	}
	fclose(spike_file);

	/* Gather the numbers of neurons to read/write from each rank */
	int *rankspikecount = 0;
	int *spikeoffsets = 0;
	int numspikestotal = 0;
	spike *allspikes = 0;
	spike *localspikes = malloc(sizeof(spike)*sr->numspikes);

	if (commrank == 0)
		rankspikecount = malloc(sizeof(int)*commsize);

	MPI_Gather(&sr->numspikes, 1, MPI_INT,
			   rankspikecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (commrank == 0) {
		spikeoffsets = len_to_offsets(rankspikecount, commsize);
		for (int i=0; i<commsize; i++)
			numspikestotal += rankspikecount[i];
		allspikes = malloc(sizeof(spike)*numspikestotal);
	}
	

	/* Read spikes back into memory and delete process-local file*/
	size_t i = 0;
	spike_file = fopen(sr->filename, "r");
	while(fscanf(spike_file, "%lf  %lu",
					&localspikes[i].time,
					&localspikes[i].neuron) != EOF) i++;
	if (i != sr->numspikes) {
		printf("Read %lu spikes, but should have been %lu spikes. Exiting.\n", i, sr->numspikes);
		exit(-1);
	}
	fclose(spike_file);
	remove(sr->filename);

	/* Gather all spikes on a single rank, sort and write (parallelize later) */
	MPI_Gatherv(localspikes, sr->numspikes, mpi_spike_type, 
				allspikes, rankspikecount, spikeoffsets,
				mpi_spike_type, 0, MPI_COMM_WORLD);

	if (commrank == 0) {
		qsort(allspikes, numspikestotal, sizeof(spike), spkcomp);
		spike_file = fopen(finalfilename, "w");
		for (int j=0; j<numspikestotal; j++) 
			fprintf(spike_file, "%lf  %lu\n", allspikes[j].time, allspikes[j].neuron);
		fclose(spike_file);
	}

	/* Clean up */
	free(sr->spikes);
	free(sr);
	free(localspikes);
	if (commrank == 0) {
		free(rankspikecount);
		free(spikeoffsets);
		free(allspikes);
	}
}


/*
 * TO BE DEPRECATED.  Allows all ranks to write their own spike file.
 */
void sr_close(spikerecord *sr)
{
	/* write any as-of-yet unsaved spikes */	
	FILE *spike_file;
	spike_file = fopen(sr->filename, sr->writemode);
	for (int i=0; i < sr-> numspikes % sr->blocksize; i++) {
		fprintf(spike_file, "%f  %lu\n", 
				sr->spikes[i].time,
				sr->spikes[i].neuron);	
	}
	fclose(spike_file);

	/* free memory */
	free(sr->spikes);
	free(sr);
}
