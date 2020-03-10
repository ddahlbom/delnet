#ifndef DELNET_H
#define DELNET_H

/*
 * -------------------- Macros --------------------
 */

#define IDX_T unsigned int
#define FLOAT_T float

#define getrandom(max1) ((rand()%(int)((max1)))) // random integer between 0 and max-1
#define unirand() (((FLOAT_T) rand()) / ((FLOAT_T) RAND_MAX + 1.0))


/*
 * -------------------- Structs --------------------
 */

typedef struct dn_vec_float_s {
	FLOAT_T *data;
	IDX_T n;
} dn_vec_float;

typedef struct dn_listnode_uint_s { 	
	unsigned int val;
	struct dn_listnode_uint_s *next;
} dn_listnode_uint;

typedef struct dn_list_uint_s {
	unsigned int count;
	struct dn_listnode_uint_s *head;
} dn_list_uint;


typedef struct dn_node_s {
	IDX_T idx_oi;
	IDX_T num_in;
	IDX_T idx_io;
	IDX_T num_out;
} dn_node;

typedef struct dn_delaynet_s {
	IDX_T *del_offsets;
	IDX_T *del_startidces;
	IDX_T *del_lens;
	IDX_T *del_sources;
	IDX_T *del_targets;
	IDX_T num_delays;
	FLOAT_T *inputs;
	FLOAT_T *outputs;
	IDX_T *inverseidx;
	IDX_T buf_len;
	FLOAT_T *delaybuf;
	IDX_T num_nodes;
	dn_node *nodes;
} dn_delaynet;


/*
 * -------------------- Functions --------------------
 */

/* lists (simple LIFO, for unsigned ints) */
dn_list_uint* 	dn_list_uint_init();
void  			dn_list_uint_push(dn_list_uint *l, unsigned int val);
unsigned int  	dn_list_uint_pop(dn_list_uint *l);

/* vector (for floats) */
void 			dn_list_uint_free(dn_list_uint *l);
char* 			dn_vectostr(dn_vec_float input);

/* make and manage a delaynet */
unsigned int* 	dn_blobgraph(unsigned int n, float p, unsigned int maxdel);
dn_delaynet* 	dn_delnetfromgraph(unsigned int *g, unsigned int n);
void 			dn_freedelnet(dn_delaynet *dn);
dn_vec_float dn_orderbuf(IDX_T which, dn_delaynet *dn);

/* control and interact with a delaynet */
void 			dn_pushoutput(FLOAT_T val, IDX_T idx, dn_delaynet *dn);
dn_vec_float 	dn_getinputvec(dn_delaynet *dn);
FLOAT_T* 		dn_getinputaddress(IDX_T idx, dn_delaynet *dn);
void 			dn_advance(dn_delaynet *dn);


#endif
