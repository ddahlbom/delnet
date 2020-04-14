#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
//#include <mpi.h> 

#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
//#include "/usr/include/mpich/mpi.h"
#include "delnetmpi.h"
#include "simutilsmpi.h"
#include "simkernelsmpi.h"
#include "paramutils.h"
#include "spkrcd.h"

#define DEBUG 0

/*************************************************************
 *  Functions
 *************************************************************/


/* -------------------- Parameter Setting -------------------- */
void su_mpi_setdefaultmparams(su_mpi_modelparams *p)
{
	p->fs = 1000.0;
	p->num_neurons = 1000;
	p->p_contact = 0.1;
	p->p_exc = 0.8;
	p->maxdelay = 20.0;
	p->tau_pre = 0.02;
	p->tau_post = 0.02;
	p->a_pre = 0.12;
	p->a_post = 0.1;
	p->synmax = 10.0;
	p->w_exc  = 6.0;
	p->w_inh = -5.0;
}


void su_mpi_readmparameters(su_mpi_modelparams *p, char *filename)
{
	paramlist *pl = pl_readparams(filename);	

	p->fs = pl_getvalue(pl, "fs");
	p->p_contact = pl_getvalue(pl, "p_contact");
	p->p_exc = pl_getvalue(pl, "p_exc");
	p->num_neurons = pl_getvalue(pl, "num_neurons");
	p->tau_pre = pl_getvalue(pl, "tau_pre");
	p->tau_post = pl_getvalue(pl, "tau_post");
	p->a_pre = pl_getvalue(pl, "a_pre");
	p->a_post = pl_getvalue(pl, "a_post");
	p->synmax = pl_getvalue(pl, "synmax");
	p->w_exc  = pl_getvalue(pl, "w_exc");
	p->w_inh = pl_getvalue(pl, "w_inh");
	p->maxdelay = pl_getvalue(pl, "maxdelay");

	pl_free(pl);
}


void su_mpi_readtparameters(su_mpi_trialparams *p, char *filename)
{
	paramlist *pl = pl_readparams(filename);
	
	p->fs = pl_getvalue(pl, "fs");
	p->dur = pl_getvalue(pl, "dur");
	p->lambda = pl_getvalue(pl, "lambda");
	p->randspikesize = pl_getvalue(pl, "randspikesize");
	p->randinput = (bool) pl_getvalue(pl, "randinput");
	p->inhibition = (bool) pl_getvalue(pl, "inhibition");
	p->numinputs = (unsigned int) pl_getvalue(pl, "numinputs");
	p->inputidcs = NULL;

	pl_free(pl);
}


void su_mpi_printmparameters(su_mpi_modelparams p)
{
	/* print trial parameters */
	printf("----------------------------------------\n");
	printf("Sampling Frequency: \t%f\n", p.fs);
	printf("Number of nodes: \t%d\n", (int) p.num_neurons);
	printf(" 	Excitatory: \t%d\n", (int) (p.num_neurons*p.p_exc));
	printf(" 	Inhibitory: \t%d\n", (int) (p.num_neurons-p.num_neurons*p.p_exc) );
	printf("Probability of contact:\t%f\n", p.p_contact);
	printf("tau_pre:\t\t%f\n", p.tau_pre);
	printf("A_pre:\t\t\t%f\n", p.a_pre);
	printf("tau_post:\t\t%f\n", p.tau_post);
	printf("A_post:\t\t\t%f\n", p.a_post);
	printf("Max synapse strength:\t%lf\n", p.synmax);
	printf("Exc syn strength:\t%lf\n", p.w_exc);
	printf("Inh syn strength:\t%lf\n", p.w_inh);
	printf("Max delay (ms):\t\t%lf\n", p.maxdelay);
	printf("----------------------------------------\n");
}


void su_mpi_analyzeconnectivity(unsigned int *g, unsigned int n,
							unsigned int n_exc, FLOAT_T fs)
{
	size_t i, j;
	double count = 0;
	double cumdur = 0;
	for (i=0; i<n_exc; i++)
	for (j=0; j<n; j++) {
		cumdur += g[i*n+j];
		count += g[i*n+j] != 0 ? 1 : 0.0;
	}
	printf("Average delay line duration (exc): %f (ms)\n", (cumdur/count)*(1000.0/fs) );
	for (i=n_exc; i<n; i++)
	for (j=0; j<n; j++) {
		count += g[i*n+j] != 0 ? 1 : 0.0;
	}
	printf("Average connections per neuron: %f\n", count/((double) n));
}


/* -------------------- Initialization Functions -------------------- */
void su_mpi_neuronset(su_mpi_neuron *n, FLOAT_T v, FLOAT_T u, FLOAT_T a, FLOAT_T d)
{
	n->v = v;
	n->u = u;
	n->a = a;
	n->d = d;
}

/* -------------------- Graph Generation -------------------- */


unsigned int *su_mpi_iblobgraph(su_mpi_modelparams *p)
{

	unsigned int *g, n, n_exc, maxdelay_n;
	size_t i, j;
	double thresh;

	n = p->num_neurons;
	n_exc = (p->num_neurons*p->p_exc);
	thresh = p->p_contact * ((float) n)/((float) n_exc);
	maxdelay_n = (p->maxdelay/1000.0) * p->fs; // since delay in ms


	g = dn_mpi_blobgraph(n, p->p_contact, maxdelay_n);
	for (i=n_exc; i<n; i++) { 			// only last 200 rows
		for (j=0; j<n_exc; j++) { 				
			if (unirand() < thresh) 
				g[i*n+j] = 1;
			else
				g[i*n+j] = 0;
		}
		for (j=n_exc; j<n; j++) 
			g[i*n+j] = 0;
	}
	return g;
}



/* Functions for running simulations */
void su_mpi_runstdpmodel(su_mpi_model_l *m, su_mpi_trialparams tp, FLOAT_T *input, size_t inputlen,
					spikerecord *sr, int commrank, int commsize, bool profiling)
{

	double gettinginputs, updatingsyntraces, updatingneurons, spikechecking,
			updatingneutraces, updatingsynstrengths, pushingoutput,
			advancingbuffer;

	gettinginputs 		 = 0;
	updatingsyntraces 	 = 0;
	updatingneurons 	 = 0;
	spikechecking 		 = 0;
	updatingneutraces 	 = 0;
	updatingsynstrengths = 0;
	pushingoutput 		 = 0;
	advancingbuffer 	 = 0;

	/* derived params -- trim later, maybe cruft */
	IDX_T n_l = m->dn->num_nodes_l;
	FLOAT_T dt = 1.0/tp.fs;
	IDX_T numsteps = tp.dur/dt;
	clock_t t_start=clock(), t_finish;

	/* local state for simulation */
	FLOAT_T *neuroninputs, *neuronoutputs; 
	FLOAT_T *nextrand = malloc(sizeof(FLOAT_T)*n_l);
	neuroninputs = calloc(n_l, sizeof(FLOAT_T));
	neuronoutputs = calloc(n_l, sizeof(FLOAT_T));
	unsigned long int numspikes = 0, numrandspikes = 0;
	//FLOAT_T offdur = 1.0; // <--- get rid of this after testing!

	/* initialize random input states */
	for(size_t i=0; i<n_l; i++) nextrand[i] = sk_mpi_expsampl(tp.lambda);


	for (size_t i=0; i<numsteps; i++) {

		/* ---------- calculate time update ---------- */
		FLOAT_T t = dt*i;
		if (i%1000 == 0)
			printf("Time: %f\n", t);


		/* ---------- get delay outputs (neuron inputs) from buffer ---------- */
		if (profiling) t_start = clock();

		sk_mpi_getinputs(neuroninputs, m->dn, m->synapses);
		numrandspikes += sk_mpi_poisnoise(neuroninputs, nextrand, t, n_l, &tp);

		if (profiling) {
			t_finish = clock();
			gettinginputs += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}

		/* put in forced input */
		//if (t < tp.dur - offdur) {
		for (size_t k=0; m->nodeoffset + k < tp.numinputs; k++) 	// (num forced inputs...)
			neuroninputs[k] += input[ i % inputlen ];
		//}

		/* ---------- update neuron state ---------- */
		if (profiling) t_start = clock();

		sk_mpi_updateneurons(m->neurons, neuroninputs, n_l, &tp);

		if (profiling) {
			t_finish = clock();
			updatingneurons += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}

		/* ---------- calculate neuron outputs ---------- */
		if (profiling) t_start = clock();

		numspikes += sk_mpi_checkspiking(m->neurons, neuronoutputs, n_l, t,
											sr, m->dn->nodeoffset);

		if (profiling) {
			t_finish = clock();
			spikechecking += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- push the neuron output into the buffer ---------- */
		if (profiling) t_start = clock();

		for (size_t k=0; k<n_l; k++)
			dn_mpi_pushoutput(neuronoutputs[k], k, m->dn); // node outputs into delnet 

		if (profiling) {
			t_finish = clock();
			pushingoutput += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- update synapse traces ---------- */
		if (profiling) t_start = clock();

		//su_mpi_updatesynapsetraces(traces_syn, spike_pre, dn, offsets, dt, &p);
		sk_mpi_updatesynapsetraces(m->traces_syn, m->dn->outputs, m->dn, dt, &m->p);

		if (profiling) {
			t_finish = clock();
			updatingsyntraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}



		/* ---------- update neuron traces ---------- */
		if (profiling) t_start = clock();

		sk_mpi_updateneurontraces(m->traces_neu, neuronoutputs, n_l, dt, &m->p);

		if (profiling) {
			t_finish = clock();
			updatingneutraces += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* ---------- update synapses ---------- */
		if (profiling) t_start = clock();

		sk_mpi_updatesynapses(m->synapses, m->traces_syn, m->traces_neu, neuronoutputs,
							m->dn, dt, &m->p);

		if (profiling) {
			t_finish = clock();
			updatingsynstrengths += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}


		/* advance the buffer */
		if (profiling) t_start = clock();

		dn_mpi_advance(m->dn);

		if (profiling) {
			t_finish = clock();
			advancingbuffer += ((double)(t_finish - t_start))/CLOCKS_PER_SEC;
		}
	}


	/* -------------------- Performance Analysis -------------------- */
	if (profiling) {
		printf("----------------------------------------\n");
		printf("Random input rate: %g\n", ((double) numrandspikes) / (((double) n_l)*tp.dur) );
		printf("Firing rate: %g\n", ((double) numspikes) / (((double) n_l)*tp.dur) );
		double cycletime, cumtime = 0.0;

		printf("----------------------------------------\n");

		cycletime = 1000.0*gettinginputs/numsteps;
		cumtime += cycletime;
		printf("Getting inputs:\t\t %f (ms)\n", cycletime);

		cycletime = 1000.0*updatingsyntraces/numsteps;
		cumtime += cycletime;
		printf("Update syntraces:\t %f (ms)\n", cycletime);

		cycletime = 1000.0*updatingneurons/numsteps;
		cumtime += cycletime;
		printf("Update neurons:\t\t %f (ms)\n", cycletime);

		cycletime = 1000.0*spikechecking/numsteps;
		cumtime += cycletime;
		printf("Check spiked:\t\t %f (ms)\n", cycletime);

		cycletime = 1000.0*pushingoutput/numsteps;
		cumtime += cycletime;
		printf("Pushing buffer:\t\t %f (ms)\n", cycletime);

		cycletime = 1000.0*updatingsynstrengths/numsteps;
		cumtime += cycletime;
		printf("Updating synapses:\t %f (ms)\n", cycletime);

		cycletime = 1000.0*advancingbuffer/numsteps;
		cumtime += cycletime;
		printf("Advancing buffer:\t %f (ms)\n", cycletime);

		printf("Total cycle time:\t %f (ms)\n", cumtime);
		printf("\nTime per second: \t %f (ms)\n", cumtime*m->p.fs);
	}

	free(neuroninputs);
	free(neuronoutputs);
	free(nextrand);
}


/* make models */
su_mpi_model_l *su_mpi_izhiblobstdpmodel(char *mparamfilename, int commrank, int commsize)
{
	unsigned int *graph = 0, n, n_exc, i;
	su_mpi_model_l *m = malloc(sizeof(su_mpi_model_l));


	/* default neuron params (Izhikevich RS and FS) */
	FLOAT_T g_v_default = -65.0;
	FLOAT_T g_u_default = -13.0;

	FLOAT_T g_a_exc  = 0.02;
	FLOAT_T g_d_exc  = 8.0;

	FLOAT_T g_a_inh  = 0.1;
	FLOAT_T g_d_inh  = 2.0;

	/* set up delnet framework */
	su_mpi_readmparameters(&m->p, mparamfilename);

	n = m->p.num_neurons;
	n_exc = (unsigned int) ((double) n * m->p.p_exc);
	size_t maxnode = dn_mpi_maxnode(commrank, commsize, n);
	size_t nodeoffset =  dn_mpi_nodeoffset(commrank, commsize, n);
	m->commrank = commrank;
	m->commsize = commsize;
	m->maxnode = maxnode;
	m->nodeoffset = nodeoffset;

	/* make sure all using the same graph */
	if (commrank == 0) {
		srand(1);
		graph = su_mpi_iblobgraph(&m->p);
		if (commsize > 1) {
			MPI_Request *sendReq = malloc(sizeof(MPI_Request) * (commsize-1));
			MPI_Request *recvReq = malloc(sizeof(MPI_Request) * (commsize-1));
			MPI_Status *recvStatus = malloc(sizeof(MPI_Status) * (commsize-1));

			/* send graph to other processes */
			for (int k=1; k<commsize; k++) {
				//MPI_Send(graph, n*n, MPI_UNSIGNED, k, 0, MPI_COMM_WORLD, &sendReq[k-1] );
				if (DEBUG) printf("Sending graph from rank 0 to rank %d\n", k);
				MPI_Send(graph, n*n, MPI_UNSIGNED, k, 0, MPI_COMM_WORLD);
				if (DEBUG) printf("Sent graph from rank 0 to rank %d\n", k);
			}

			/* wait for processes to be received */
			//for (int k=1; k<commsize; k++) {
			//	MPI_Wait(&recvReq[k-1], &recvStatus[k-1]);
			//}
			free(sendReq);
			free(recvReq);
			free(recvStatus);
		}
	} else {
		MPI_Status recvStatus;
		graph = malloc(sizeof(unsigned int)*n*n);
		if (DEBUG) printf("Waiting for graph on rank %d from rank 0\n", commrank);
		MPI_Recv(graph, n*n, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &recvStatus);
		if (DEBUG) printf("Received graph on rank %d from rank 0\n", commrank);
	}

	if (DEBUG) printf("Making delnet on process %d\n", commrank);
	m->dn = dn_mpi_delnetfromgraph(graph, n, commrank, commsize);
	if (DEBUG) printf("Made delnet on process %d\n", commrank);


	/* set up state for simulation */
	if (DEBUG) printf("Allocating state on rank %d\n", commrank);
	su_mpi_neuron *neurons  = malloc(sizeof(su_mpi_neuron)*maxnode);
	FLOAT_T *traces_neu 	= calloc(maxnode, sizeof(FLOAT_T));
	FLOAT_T *traces_syn; 	
	FLOAT_T *synapses; 		
	IDX_T numsyn_tot=0, numsyn_exc;

	for (i=0; i<maxnode; i++) {
		if (nodeoffset + i < n_exc)
			su_mpi_neuronset(&neurons[i], g_v_default, g_u_default, g_a_exc, g_d_exc);
		else
			su_mpi_neuronset(&neurons[i], g_v_default, g_u_default, g_a_inh, g_d_inh);
	}

	// Find a better way than this to find number of excitatory synapses
	for (i=0; i<n_exc; i++) {
		numsyn_tot += m->dn->nodes[i].num_in;
	}
	numsyn_exc = numsyn_tot;
	for (i = n_exc; i<n && i<maxnode; i++) {
		numsyn_tot += m->dn->nodes[i].num_in;
	}

	//traces_syn = calloc(numsyn_tot, sizeof(FLOAT_T));		
	//synapses = calloc(numsyn_tot, sizeof(FLOAT_T));
	traces_syn = calloc(m->dn->numlinesin_l, sizeof(FLOAT_T));		
	synapses = calloc(m->dn->numlinesin_l, sizeof(FLOAT_T));
	
	/* initialize synapse weights */
	if (DEBUG) printf("Initializing synapses on rank %d\n", commrank);
	//for (i=0; i < numsyn_exc; i++)
	//	synapses[m->dn->destidx_g[i]] = m->p.w_exc;
	//for (; i < numsyn_tot; i++)
	//	synapses[m->dn->destidx_g[i]] = m->p.w_inh;
	unsigned int i_g;	
	for (i=0; i < m->dn->numlinesin_l; i++) {
		i_g = i + m->dn->lineoffset_out; 	// <----- confirm this (out or in?)
		synapses[i] = m->dn->sourceidx_g[i_g] < numsyn_exc ? m->p.w_exc : m->p.w_inh;
	}

	m->numinputneurons = 100; 	// <- refactor out -- now in trial params
	m->neurons = neurons;
	m->traces_neu = traces_neu;
	m->traces_syn = traces_syn;
	m->synapses = synapses;

	if (DEBUG) printf("About to free graph on rank %d\n", commrank);
	free(graph);

	return m;
}


/* loading and freeing models */

void su_mpi_savemodel_l(su_mpi_model_l *m, char *filename)
{
	FILE *f = fopen(filename, "wb");
	
	dn_mpi_save(m->dn, f);

	fwrite(&m->numinputneurons, sizeof(IDX_T), 1, f);
	fwrite(&m->commrank, sizeof(int), 1, f);
	fwrite(&m->commsize, sizeof(int), 1, f);
	fwrite(&m->maxnode, sizeof(size_t), 1, f);
	fwrite(&m->nodeoffset, sizeof(size_t), 1, f);
	fwrite(&m->numsyn, sizeof(IDX_T), 1, f);
	fwrite(&m->p, sizeof(su_mpi_modelparams), 1, f);
	fwrite(m->neurons, sizeof(su_mpi_neuron), m->dn->num_nodes_l, f);
	fwrite(m->traces_neu, sizeof(FLOAT_T), m->dn->num_nodes_l, f);
	fwrite(m->traces_syn, sizeof(FLOAT_T), m->dn->numlinesin_l, f);
	fwrite(m->synapses, sizeof(FLOAT_T), m->dn->numlinesin_l, f);

	fclose(f);
}

su_mpi_model_l *su_mpi_loadmodel_l(char *filename)
{
	su_mpi_model_l *m = malloc(sizeof(su_mpi_model_l));
	size_t loadsize;
	FILE *f = fopen(filename, "rb");
	
	m->dn = dn_mpi_load(f);

	loadsize = fread(&m->numinputneurons, sizeof(IDX_T), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->commrank, sizeof(int), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->commsize, sizeof(int), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->maxnode, sizeof(size_t), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->nodeoffset, sizeof(size_t), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->numsyn, sizeof(IDX_T), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->p, sizeof(su_mpi_modelparams), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	m->neurons = malloc(sizeof(su_mpi_neuron)*m->dn->num_nodes_l);
	loadsize = fread(m->neurons, sizeof(su_mpi_neuron), m->dn->num_nodes_l, f);
	if (loadsize != m->dn->num_nodes_l) { printf("Failed to load model.\n"); exit(-1); }

	m->traces_neu = malloc(sizeof(FLOAT_T)*m->dn->num_nodes_l);
	loadsize = fread(m->traces_neu, sizeof(FLOAT_T), m->dn->num_nodes_l, f);
	if (loadsize != m->dn->num_nodes_l) { printf("Failed to load model.\n"); exit(-1); }

	m->traces_syn = malloc(sizeof(FLOAT_T)*m->dn->numlinesin_l);
	loadsize = fread(m->traces_syn, sizeof(FLOAT_T), m->dn->numlinesin_l, f);
	if (loadsize != m->dn->numlinesin_l) { printf("Failed to load model.\n"); exit(-1); }

	m->synapses = malloc(sizeof(FLOAT_T)*m->dn->numlinesin_l);
	loadsize = fread(m->synapses, sizeof(FLOAT_T), m->dn->numlinesin_l, f);
	if (loadsize != m->dn->numlinesin_l) { printf("Failed to load model.\n"); exit(-1); }

	fclose(f);

	return m;
}

void su_mpi_freemodel_l(su_mpi_model_l *m) {
	dn_mpi_freedelnet(m->dn);
	free(m->neurons);
	free(m->traces_neu);
	free(m->traces_syn);
	free(m->synapses);
	free(m);
}

