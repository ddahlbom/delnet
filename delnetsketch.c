#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX_T unsigned int
#define FLOAT_T float

/*
 * -------------------- Structs --------------------
 */

typedef struct vec_s {
	FLOAT_T *data;
	IDX_T n;
} vec;

typedef struct node_s {
	IDX_T idx_oi;
	IDX_T num_in;
	IDX_T idx_io;
	IDX_T num_out;
} node;

typedef struct delay_s {
	IDX_T offset;
	IDX_T startidx;
	IDX_T len;
	IDX_T source;
	IDX_T target;
} delay;

typedef struct delaynet_s {
	IDX_T num_delays;
	delay *delays;
	FLOAT_T *inputs;
	FLOAT_T *outputs;
	IDX_T *inverseidx;
	IDX_T buf_len;
	FLOAT_T *delaybuf;
	IDX_T num_nodes;
	node *nodes;
} delaynet;


/*
 * -------------------- Functions --------------------
 */
void pushoutput(FLOAT_T val, IDX_T idx, delaynet *dn) 
{
	IDX_T i1, i2, k;

	i1 = dn->nodes[idx].idx_io;
	i2 = i1 + dn->nodes[idx].num_out;

	for (k = i1; k < i2; k++)
		dn->inputs[k] = val;
}


/* No getinputs()... would need to return vector */
FLOAT_T *getinputaddress(IDX_T idx, delaynet *dn)
{
	return &dn->outputs[dn->nodes[idx].idx_oi];
}


void advance(delaynet *dn)
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

vec orderbuf(delay *d, FLOAT_T *delbuf) {
	IDX_T k, n, idx;
	vec output;
	
	n = d->len;
	output.n = n;
	output.data = malloc(sizeof(FLOAT_T) * n);

	for (k=0; k<n; k++) {
		idx = d->startidx + ((d->offset+k) % d->len);
		output.data[n-k-1] = delbuf[idx];
	}
	return output;
}

char *vectostr(vec input) {
	int k;
	char *output;
	output = malloc(sizeof(char)*(input.n+1));
	output[input.n] = "\0";
	for(k=0; k < input.n; k++) {
		output[k] = 
	}

}


/*
 * -------------------- Main Test --------------------
 */
int main()
{
			

	return 0;
}
