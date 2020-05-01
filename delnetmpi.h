#ifndef DELNETMPI_H
#define DELNETMPI_H

#include <stdio.h>

/*
 * -------------------- Macros --------------------
 */

#define IDX_T unsigned int
#define FLOAT_T double

#define getrandom(max1) ((rand()%(int)((max1)))) // random integer between 0 and max-1
#define unirand() (((FLOAT_T) rand()) / ((FLOAT_T) RAND_MAX + 1.0))

#define MAXBUFFERLEN 8 

/*
 * -------------------- Structs --------------------
 */

typedef struct dn_mpi_vec_float_s {
	FLOAT_T *data;
	IDX_T n;
} dn_mpi_vec_float;

typedef struct dn_mpi_listnode_uint_s { 	
	unsigned int val;
	struct dn_mpi_listnode_uint_s *next;
} dn_mpi_listnode_uint;

typedef struct dn_mpi_list_uint_s {
	unsigned int count;
	struct dn_mpi_listnode_uint_s *head;
} dn_mpi_list_uint;


typedef struct dn_mpi_eventbuffer_s {
	int buffer[MAXBUFFERLEN];
	unsigned int bufferlen;
} dn_mpi_eventbuffer;


typedef struct dn_mpi_node_s {
	IDX_T idx_outbuf; 	// output of delnet to input of nodes
	IDX_T num_in;
	IDX_T idx_inbuf; 	// output of nodes to input of delnet
	IDX_T num_out;
} dn_mpi_node;

typedef struct dn_mpi_delaynet_s {
	/* size info */
	IDX_T num_nodes_g;
	IDX_T num_nodes_l;
	IDX_T nodeoffset;
	IDX_T numlinesout_l;
	IDX_T numlinesin_l;
	IDX_T numlines_g;
	IDX_T lineoffset_in;
	IDX_T lineoffset_out;
	IDX_T buf_len;
	int commrank;
	int commsize;

	/* pointers */
	FLOAT_T *delaybuf;
	FLOAT_T *inputs;
	FLOAT_T *outputs;
	FLOAT_T *outputs_unsorted;
	dn_mpi_node *nodes;
	IDX_T *destidx_g;
	IDX_T *sourceidx_g;
	//IDX_T *del_offsets;
	//IDX_T *del_startidces;
	//IDX_T *del_lens;
	IDX_T *del_sources;
	IDX_T *del_targets;

	/* MPI revisions */
	unsigned int *outblocksizes;
	unsigned int *outblockoffsets;

	/* non-buffered revision */
	dn_mpi_eventbuffer *ebs;


} dn_mpi_delaynet;


/*
 * -------------------- Functions --------------------
 */

/* MPI utils */
size_t dn_mpi_maxnode(int rank, int commsize, size_t numpoints);
size_t dn_mpi_nodeoffset(int rank, int commsize, size_t numpoints);
void dn_mpi_syncoutputs(dn_mpi_delaynet *dn);

/* lists (simple LIFO, for unsigned ints) */
dn_mpi_list_uint* 	dn_mpi_list_uint_init();
void  			dn_mpi_list_uint_push(dn_mpi_list_uint *l, unsigned int val);
unsigned int  	dn_mpi_list_uint_pop(dn_mpi_list_uint *l);

/* vector (for floats) */
void 			dn_mpi_list_uint_free(dn_mpi_list_uint *l);
char* 			dn_mpi_vectostr(dn_mpi_vec_float input);

/* make and manage a delaynet */
unsigned int* 	dn_mpi_blobgraph(unsigned int n, float p, unsigned int maxdel);
dn_mpi_delaynet *dn_mpi_delnetfromgraph(unsigned int *g, unsigned int n, 
											int commsize, int commrank);
void 			dn_mpi_freedelnet(dn_mpi_delaynet *dn);
dn_mpi_vec_float dn_mpi_orderbuf(IDX_T which, dn_mpi_delaynet *dn);

/* control and interact with a delaynet */
void 			dn_mpi_pushoutput(FLOAT_T val, IDX_T idx, dn_mpi_delaynet *dn);
dn_mpi_vec_float 	dn_mpi_getinputvec(dn_mpi_delaynet *dn);
FLOAT_T* 		dn_mpi_getinputaddress(IDX_T idx, dn_mpi_delaynet *dn);
void 			dn_mpi_advance(dn_mpi_delaynet *dn);

/* save and load checkpoints (with buffers), network (no buffers)*/
void dn_mpi_savecheckpt(dn_mpi_delaynet *dn, FILE *stream);
dn_mpi_delaynet *dn_mpi_loadcheckpt(FILE *stream);
void dn_mpi_save(dn_mpi_delaynet *dn, FILE *stream);
dn_mpi_delaynet *dn_mpi_load(FILE *stream);

/* nonbuffered event handling */
void dn_mpi_pushevent(dn_mpi_eventbuffer *eb);
FLOAT_T dn_mpi_advancebuffer(dn_mpi_eventbuffer *eb);

#endif
