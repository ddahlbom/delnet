#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

typedef struct dn_delay_s {
	IDX_T offset;
	IDX_T startidx;
	IDX_T len;
	IDX_T source;
	IDX_T target;
} dn_delay;

typedef struct dn_delaynet_s {
	IDX_T num_delays;
	dn_delay *delays;
	FLOAT_T *inputs;
	FLOAT_T *outputs;
	IDX_T *inverseidx;
	IDX_T buf_len;
	FLOAT_T *delaybuf;
	IDX_T num_nodes;
	dn_node *nodes;
} dn_delaynet;


/*
 * -------------------- dd Functions --------------------
 */

dn_list_uint *dn_list_uint_init() {
	dn_list_uint *newlist;
	newlist = malloc(sizeof(dn_list_uint));
	newlist->count = 0;
	newlist->head = NULL; 

	return newlist;
}

void dn_list_uint_push(dn_list_uint *l, unsigned int val) {
	dn_listnode_uint *newnode;
	newnode = malloc(sizeof(dn_listnode_uint));
	newnode->val = val;
	newnode->next = l->head;
	l->head = newnode;
	l->count += 1;
}

unsigned int dn_list_uint_pop(dn_list_uint *l) {
	unsigned int val;
	dn_listnode_uint *temp;

	if (l->head != NULL) {
		val = l->head->val;
		temp = l->head;
		l->head = l->head->next;
		free(temp);
		l->count -= 1;
	}
	else {
		printf("Tried to pop empty list! Returning 0...\n");
		val = 0;
	}
	return val;
}

void dn_list_uint_free(dn_list_uint *l) {
	while (l->head != NULL) {
		dn_list_uint_pop(l);
	}
	free(l);
}


dn_vec_float dn_orderbuf(dn_delay *d, FLOAT_T *delbuf) {
	IDX_T k, n, idx;
	dn_vec_float output;
	
	n = d->len;
	output.n = n;
	output.data = malloc(sizeof(FLOAT_T) * n);

	for (k=0; k<n; k++) {
		idx = d->startidx + ((d->offset+k) % d->len);
		output.data[n-k-1] = delbuf[idx];
	}
	return output;
}

char *dn_vectostr(dn_vec_float input) {
	int k;
	char *output;
	output = malloc(sizeof(char)*(input.n+1));
	output[input.n] = '\0';
	for(k=0; k < input.n; k++) {
		output[k] = input.data[k] == 0.0 ? '-' : '*';
	}
	return output;
}


/*
 * -------------------- delnet Functions --------------------
 */

void dn_pushoutput(FLOAT_T val, IDX_T idx, dn_delaynet *dn) 
{
	IDX_T i1, i2, k;

	i1 = dn->nodes[idx].idx_io;
	i2 = i1 + dn->nodes[idx].num_out;

	for (k = i1; k < i2; k++)
		dn->inputs[k] = val;
}


/* No getinputs()... would need to return vector */
FLOAT_T *dn_getinputaddress(IDX_T idx, dn_delaynet *dn)
{
	return &dn->outputs[dn->nodes[idx].idx_oi];
}


void dn_advance(dn_delaynet *dn)
{
	IDX_T k;

	/* load input */	
	for(k=0; k < dn->num_delays; k++) {
		dn->delaybuf[dn->delays[k].startidx + dn->delays[k].offset] =
															dn->inputs[k];
	}

	/* advance buffer */
	for(k=0; k < dn->num_delays; k++) {
		dn->delays[k].offset = (dn->delays[k].offset + 1) % dn->delays[k].len;
	}

	/* pull output */
	for (k=0; k < dn->num_delays; k++) {
		dn->outputs[dn->inverseidx[k]] =
				dn->delaybuf[dn->delays[k].startidx+dn->delays[k].offset];
	}
}

unsigned int *dn_blobgraph(unsigned int n, float p, unsigned int maxdel) {
	unsigned int count = 0;
	unsigned int *delmat;
	unsigned int i, j;
	delmat = malloc(sizeof(int)*n*n);

	for (i=0; i<n; i++) 
	for (j=0; j<n; j++) {
		if (unirand() < p && i != j) {
			delmat[i*n + j] = getrandom(maxdel) + 1;
			count += 1;
		}
		else 
			delmat[i*n + j] = 0;
	}

	printf("Percent non-zero: %f\n", ((float) count) / ((float)n*(float)n));
	return delmat;
}


dn_delaynet *dn_delnetfromgraph(unsigned int *g, unsigned int n) {
	unsigned int i, j, delcount, startidx;
	unsigned int deltot, numlines;
	dn_delaynet *dn;
	dn_list_uint **nodes_in;

	dn = malloc(sizeof(dn_delaynet));
	nodes_in = malloc(sizeof(dn_list_uint)*n);
	for (i=0; i<n; i++)
		nodes_in[i] = dn_list_uint_init();

	deltot = 0;
	numlines = 0;
	for (i=0; i<n*n; i++) {
		deltot += g[i];
		numlines += g[i] != 0 ? 1 : 0;
	}
	dn->num_delays = numlines;
	dn->buf_len = deltot;
	dn->num_nodes = n;

	dn->delaybuf = calloc(deltot, sizeof(FLOAT_T));
	dn->inputs = calloc(numlines, sizeof(FLOAT_T));
	dn->outputs = calloc(numlines, sizeof(FLOAT_T));

	dn->delays = malloc(sizeof(dn_delay)*numlines);
	dn->nodes = malloc(sizeof(dn_node)*n);

	/* init nodes */
	for (i=0; i<n; i++) {
		dn->nodes[i].idx_oi = 0;
		dn->nodes[i].num_in = 0;
		dn->nodes[i].idx_io = 0;
		dn->nodes[i].num_out = 0;
	}

	/* work through graph */
	delcount = 0;
	startidx = 0;
	for (i = 0; i<n; i++)
	for (j = 0; j<n; j++) {
		if (g[i*n + j] != 0) {
			dn_list_uint_push(nodes_in[j], i);

			dn->delays[delcount].offset = 0;
			dn->delays[delcount].startidx = startidx;
			dn->delays[delcount].len = g[i*n + j];
			dn->delays[delcount].source = i;
			dn->delays[delcount].target = j;
			dn->nodes[i].num_out += 1;

			startidx += g[i*n +j];
			delcount += 1;
		}
	}
	
	/* work out rest of index arithmetic */
	unsigned int *num_outputs, *in_base_idcs;
	num_outputs = calloc(n, sizeof(unsigned int));
	in_base_idcs = calloc(n, sizeof(unsigned int));
	for (i=0; i<n; i++) {
		num_outputs[i] = dn->nodes[i].num_out;
		for (j=0; j<i; j++)
			in_base_idcs[i] += num_outputs[j]; 	// check logic here
	}

	unsigned int idx = 0;
	for (i=0; i<n; i++) {
		dn->nodes[i].num_in = nodes_in[i]->count;
		dn->nodes[i].idx_oi = idx;
		idx += dn->nodes[i].num_in;
		dn->nodes[i].idx_io = in_base_idcs[i];
	}

	unsigned int *num_inputs, *out_base_idcs, *out_counts, *inverseidces;
	num_inputs = calloc(n, sizeof(unsigned int));
	out_base_idcs = calloc(n, sizeof(unsigned int));
	out_counts = calloc(n, sizeof(unsigned int));
	for (i=0; i<n; i++) {
		num_inputs[i] = dn->nodes[i].num_in;
		for (j=0; j<n; j++)
			out_base_idcs[i] += num_inputs[j]; // check logic here
	}

	inverseidces = calloc(numlines, sizeof(unsigned int));
	for (i=0; i<n; i++) {
		inverseidces[i] = out_base_idcs[dn->delays[i].target] + 
						  out_counts[dn->delays[i].target];
		out_counts[dn->delays[i].target] += 1;
	}
	dn->inverseidx = inverseidces;

	/* Clean up */
	for (i=0; i<n; i++)
		dn_list_uint_free(nodes_in[i]);
	free(nodes_in);
	free(num_outputs);
	free(in_base_idcs);
	free(num_inputs);
	free(out_base_idcs);
	free(out_counts);

	return dn;
}

void dn_freedelnet(dn_delaynet *dn) {
	free(dn->delays);
	free(dn->inputs);
	free(dn->outputs);
	free(dn->inverseidx);
	free(dn->delaybuf);
	free(dn->nodes);
	free(dn);
}

/*
 * -------------------- Main Test --------------------
 */
int main()
{
			
	dn_vec_float myvector;
	char *mystr;
	unsigned int *delmat;

	/* Test vector */
	myvector.data = malloc(sizeof(FLOAT_T)*10);
	myvector.n = 10;
	
	for (int k = 0; k < 10; k++) {
		myvector.data[k] = k;
	}

	/* Test string buffer stuff */
	mystr = dn_vectostr(myvector);
	printf("%s\n", mystr);

	/* Test blob graph */
	unsigned int dim = 1000;
	delmat = dn_blobgraph(dim,0.1,20);

	/* Test delaynet */
	dn_delaynet *dn = dn_delnetfromgraph(delmat, dim);
	dn_freedelnet(dn);

	/* Test list */
	dn_list_uint *l = dn_list_uint_init();
	dn_list_uint_push(l, 9);
	dn_list_uint_push(l, 8);
	dn_list_uint_push(l, 7);

	while (l->count > 0) 
		printf("%u\n", dn_list_uint_pop(l));


	/* Clean up */
	dn_list_uint_free(l);
	free(mystr);
	free(myvector.data);
	free(delmat);

	return 0;
}
