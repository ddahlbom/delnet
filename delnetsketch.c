#include <stdio.h>
#include <stdlib.h>

#include "delnet.h"

/*
 * -------------------- Main Test --------------------
 */
int main()
{
	/* Test blob graph */
	// unsigned int *delmat;
	// unsigned int n = 1000;
	// delmat = dn_blobgraph(n,0.1,20);
	
	/* Test handcrafted graph */
	unsigned int n = 4;
	unsigned int delmat[] = { 0, 3, 3, 0,
							  0, 0, 3, 0,
							  0, 0, 0, 3,
							  3, 0, 0, 0 };
	unsigned int numlines = 0;
	unsigned int deltot = 0;
	unsigned int numsteps = 6;
	FLOAT_T nodevals[] = { 1.0, 1.0, 0.0, 0.0 };

	for (int i=0; i<n; i++)
	for (int j=0; j<n; j++) {
		numlines += delmat[i*n+j] == 0 ? 0 : 1;
		deltot += delmat[i*n+j];
	}

	/* Test delaynet for memory leaks */
	dn_delaynet *dn = dn_delnetfromgraph(delmat, n);
	dn_vec_float invals, linevals;
	char *dispstr;

	unsigned int i, j, k;
	for (j=0; j<numsteps; j++) {

		// push in neuron outputs into dl inputs
		for (k=0; k<n; k++) {
			dn_pushoutput(nodevals[k], k, dn);
		}
		
		// print out state
		printf("\nSTEP %u:\n", j+1);
		printf("nodevals: [");
		for (k=0; k<n-1; k++) {
			printf("%1.1f, ", nodevals[k]);	
		}
		printf("%1.1f]\n", nodevals[n-1]);
		invals = dn_getinputvec(dn);
		dispstr = dn_vectostr(invals); 
		printf("%s\n\n", dispstr);

		free(dispstr);
		free(invals.data);

		// advance
		dn_advance(dn);

		// show states
		for (k=0; k<numlines; k++) {
			linevals = dn_orderbuf(k, dn);	
			dispstr = dn_vectostr(linevals);
			printf("(%u) %s (%u)\n", dn->del_sources[k]+1,
									 dispstr,
									 dn->del_targets[k]+1);
			free(dispstr);
			free(linevals.data);
		}

		// print delaybuffer output (neuron input)
		FLOAT_T *inputaddr = dn->outputs;
		dn_vec_float bufoutput;
		bufoutput.data = dn->outputs;
		bufoutput.n = dn->num_delays;
		dispstr = dn_vectostr(bufoutput);
		printf("\n%s\n\n", dispstr);
		free(dispstr);
		
		// pull outputs
		for (k=0; k<n; k++) {
			inputaddr = dn_getinputaddress(k, dn);	
			nodevals[k] = 0.0;
			for (i=0; i < dn->nodes[k].num_in; i++) {
				nodevals[k] += *(inputaddr+i);
			}
			nodevals[k] = nodevals[k] > 1 ? 1 : 0;
		}

		printf("############################################################\n");

	}

	printf("[%u, ", dn->inverseidx[0] + 1);
	for (i=1; i<dn->num_delays-1; i++) {
		printf("%u, ", dn->inverseidx[i] + 1);
	}
	printf("%u]\n", dn->inverseidx[dn->num_delays-1] + 1);

	/* Clean up */
	dn_freedelnet(dn);
	free(invals.data);

	return 0;
}
