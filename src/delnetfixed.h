#define DNF_BUF_SIZE 15  // 2^n - 1
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
	unsigned short counts[DNF_BUF_SIZE];	
} dnf_delaybuf;


typedef struct dnf_delaynet_s {
	//dnf_node *nodes;
	idx_t numnodes;
	data_t *nodeinputbuf;
	idx_t *nodebufferoffsets;
	idx_t *numbuffers; // per node
	idx_t numbufferstotal;
	dnf_delaybuf *buffers;

	idx_t **dests; 	//destination indexed [target rank][local neuron number]
	idx_t **destoffsets;
	idx_t **destlens;
	idx_t *destlenstot; // maximum number of targets per rank

	idx_t **sendblocks;
	idx_t **recvblocks;
} dnf_delaynet;


/* -------------------- Function Declarations -------------------- */

/* buffer functions */
static inline dnf_error dnf_bufinit(dnf_delaybuf *buf, unsigned short len);
static inline dnf_error dnf_recordevent(dnf_delaybuf *buf);
static inline dnf_error dnf_bufadvance(dnf_delaybuf *buf, data_t *out);

/* delaynet interaction */
void dnf_pushevents(dnf_delaynet *dn, idx_t *eventnodes, idx_t numevents,
					int commrank, int commsize);
void dnf_advance(dnf_delaynet *dn);
data_t *dnf_getinputaddress(dnf_delaynet *dn, idx_t node);

/* delaynet generation */
dnf_delaynet *dnf_delaynetfromgraph(unsigned long *graph, unsigned long n,
									int commrank, int commsize);
void dnf_freedelaynet(dnf_delaynet *dn, int commsize);
