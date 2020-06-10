#ifndef DELNETFIXED_H
#define DELNETFIXED_H

#define DNF_BUF_SIZE 15  // 2^n - 1
#define FLOAT_T double 	// remove later
#define IDX_T unsigned long
#define data_t double
#define idx_t unsigned long
#define mpi_idx_t MPI_UNSIGNED_LONG

typedef enum dnf_error {
	DNF_SUCCESS,
	DNF_BUFFER_OVERFLOW,
	DNF_GRAPHINIT_FAIL
} dnf_error;


/* --------------------	Structures -------------------- */
/* delaynet proper */
typedef struct dnf_node_s {
	idx_t *numtargetranks;
	idx_t *numtargetsperrank;
	idx_t *targets;
	idx_t numinputs;
	idx_t inputoffset;
} dnf_node;


typedef struct dnf_delaybuf_s {
	unsigned short delaylen;
	unsigned short numstored;
	unsigned short counts[DNF_BUF_SIZE];	
} dnf_delaybuf;


typedef struct dnf_delaynet_s {
	//dnf_node *nodes;
	int commrank;
	int commsize;
	idx_t numnodes;
	idx_t nodeoffsetglobal;
	idx_t bufferoffsetglobal;

	data_t *nodeinputbuf;
	idx_t *nodebufferoffsets;
	idx_t *numbuffers; // per node
	idx_t numbufferstotal;
	idx_t *buffersourcenodes;
	dnf_delaybuf *buffers;

	idx_t **destlens;
	idx_t **destoffsets;
	idx_t **dests; 	//destination indexed [target rank][local neuron number]
	idx_t *destlenstot; // maximum number of targets per rank
	idx_t *recvlenstot; // maximum number of *incoming* targets from each rank

	idx_t **sendblocks;
	idx_t **recvblocks;
} dnf_delaynet;


/* -------------------- Function Declarations -------------------- */

/* delaynet interaction */
void dnf_pushevents(dnf_delaynet *dn, idx_t *eventnodes, idx_t numevents,
					int commrank, int commsize, double t);
void dnf_advance(dnf_delaynet *dn);
data_t *dnf_getinputaddress(dnf_delaynet *dn, idx_t node);

/* delaynet generation */
dnf_delaynet *dnf_delaynetfromgraph(unsigned long *graph, unsigned long n,
									int commrank, int commsize);
void dnf_freedelaynet(dnf_delaynet *dn);

/* IO */
void dnf_save(dnf_delaynet *dn, FILE *stream);
dnf_delaynet *dnf_load(FILE *stream);

/* buffer functions */
dnf_error dnf_bufinit(dnf_delaybuf *buf, unsigned short len);
dnf_error dnf_recordevent(dnf_delaybuf *buf, double t);
dnf_error dnf_bufadvance(dnf_delaybuf *buf, data_t *out);

/* indexing utils */
idx_t dnf_maxnode(int commrank, int commsize, idx_t numpoints);
idx_t dnf_nodeoffset(int commrank, int commsize, idx_t numpoints);
idx_t *dnf_getlens(int commsize, idx_t numpoints);
idx_t *dnf_getstartidcs(int commsize, idx_t numpoints);


#endif
