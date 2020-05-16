#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>


#ifdef __amd64__
#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#define CLOCKSPEED 3200000000
typedef unsigned long long ticks;
char perfFileName[] = "performance_data.txt";
static __inline__ ticks getticks(void)
{
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((ticks)hi << 32) | lo;
}

#else

#include <mpi.h>
#define CLOCKSPEED 512000000
typedef unsigned long long ticks;
char perfFileName[] = "/gpfs/u/home/PCP9/PCP9dhlb/barn/delnet/performance_data.txt";
char prefix[] = "/gpfs/u/home/PCP9/PCP9dhlb/scratch/";
static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((ticks)tbu0) << 32) | tbl;
}
#endif

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
	
	p->dur = pl_getvalue(pl, "dur");
	p->lambda = pl_getvalue(pl, "lambda");
	p->randspikesize = pl_getvalue(pl, "randspikesize");
	p->randinput = (bool) pl_getvalue(pl, "randinput");
	p->inhibition = (bool) pl_getvalue(pl, "inhibition");
	p->numinputs = (unsigned int) pl_getvalue(pl, "numinputs");
	p->inputmode = (unsigned int) pl_getvalue(pl, "inputmode");
	p->recordstart = pl_getvalue(pl, "recordstart");
	p->recordstop = pl_getvalue(pl, "recordstop");
	p->lambdainput = pl_getvalue(pl, "lambdainput");

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


static inline double meantime(double *vals, int n)
{
	double mean = 0;
	for (int k=0; k<n; k++) mean += vals[k];
	return mean/(double)n;
}

static inline double maxtime(double *vals, int n) 
{
	double max = 0;
	for (int k=0; k<n; k++) 
		if (vals[k] > max) max = vals[k];
	return max;
}

/* Functions for running simulations */
void su_mpi_runstdpmodel(su_mpi_model_l *m, su_mpi_trialparams tp,
							su_mpi_spike *input, size_t inputlen,
							spikerecord *sr, char *trialname,
							int commrank, int commsize, bool profiling)
{

	ticks gettinginputs, updatingsyntraces, updatingneurons, spikechecking,
			updatingneutraces, updatingsynstrengths, pushingoutput,
			advancingbuffer, ticks_start=0, ticks_finish, totalticks_start,
		  	totalticks_finish, totaltickscum=0;

	/* timing info */
	gettinginputs 		 = 0;
	updatingsyntraces 	 = 0;
	updatingneutraces 	 = 0;
	updatingneurons 	 = 0;
	spikechecking 		 = 0;
	updatingneutraces 	 = 0;
	updatingsynstrengths = 0;
	pushingoutput 		 = 0;
	advancingbuffer 	 = 0;

	/* derived params -- trim later, maybe cruft */
	IDX_T n_l = m->dn->num_nodes_l;
	FLOAT_T dt = 1.0/m->p.fs;
	IDX_T numsteps = tp.dur/dt;

	/* local state for simulation */
	FLOAT_T *neuroninputs, *neuronoutputs; 
	FLOAT_T *nextrand = malloc(sizeof(FLOAT_T)*n_l);
	//FLOAT_T nextinputtime = 0.0;
	//bool waiting = true;
	neuroninputs = calloc(n_l, sizeof(FLOAT_T));
	neuronoutputs = calloc(n_l, sizeof(FLOAT_T));
	unsigned long int numspikes = 0, numrandspikes = 0;
	FLOAT_T t;

	FILE *inputtimesfile = 0;
	char filename[MAX_NAME_LEN];
	sprintf(filename, "%s_instarttimes.txt", trialname);
	// size_t inputidx = 0;
	// unsigned int inputcounter = 0;

	//double t_local = 0.0;
	double t_max = 0.0;
	for (int i=0; i<inputlen; i++)
		if (input[i].t > t_max) t_max = input[i].t;

	if (commrank == 0)
		inputtimesfile = fopen(filename, "w");

	/* initialize random input states */
	for(size_t i=0; i<n_l; i++) nextrand[i] = sk_mpi_expsampl(tp.lambda);

	for (size_t i=0; i<numsteps; i++) {
		if (profiling) totalticks_start = getticks(); 

		/* ---------- calculate time update ---------- */
		t = dt*i;
		if (i%1000 == 0)
			printf("Time: %f\n", t);


		/* ---------- inputs ---------- */
		if (profiling) ticks_start = getticks();

		/* get inputs from delay net */
		sk_mpi_getinputs(neuroninputs, m->dn, m->synapses);

		/* put in random noise */
		numrandspikes += sk_mpi_poisnoise(neuroninputs, nextrand, t, n_l, &tp);

		/* put in forced input -- make this a function in kernels! */
		sk_mpi_forcedinput( m, input, inputlen, neuroninputs, t, dt, t_max,
							&tp, commrank, commsize, inputtimesfile ); 

		if (profiling) {
			ticks_finish = getticks(); 
			gettinginputs += (ticks_finish - ticks_start);
		}


		/* ---------- update neuron state ---------- */
		if (profiling) ticks_start = getticks();

		sk_mpi_updateneurons(m->neurons, neuroninputs, n_l, m->p.fs);

		if (profiling) {
			ticks_finish = getticks();
			updatingneurons += (ticks_finish - ticks_start);
		}


		/* ---------- calculate neuron outputs ---------- */
		if (profiling) ticks_start = getticks();

		numspikes += sk_mpi_checkspiking(m->neurons, neuronoutputs, n_l, t,
											sr, m->dn->nodeoffset,
											tp.recordstart, tp.recordstop);

		if (profiling) {
			ticks_finish = getticks();
			spikechecking += (ticks_finish - ticks_start);
		}

		/* ---------- push the neuron output into the buffer ---------- */
		if (profiling) ticks_start = getticks(); 

		for (size_t k=0; k<n_l; k++)
			dn_mpi_pushoutput(neuronoutputs[k], k, m->dn); // node outputs into delnet 
		if (profiling) {
			ticks_finish = getticks();
			pushingoutput += (ticks_finish - ticks_start);
		}


		/* ---------- update synapse traces ---------- */
		if (profiling) ticks_start = getticks();

		sk_mpi_updatesynapsetraces(m->traces_syn, m->dn->outputs, m->dn, dt, &m->p);

		if (profiling) {
			ticks_finish = getticks();
			updatingsyntraces += (ticks_finish - ticks_start);
		}


		/* ---------- update neuron traces ---------- */
		if (profiling) ticks_start = getticks();

		sk_mpi_updateneurontraces(m->traces_neu, neuronoutputs, n_l, dt, &m->p);

		if (profiling) {
			ticks_finish = getticks();
			updatingneutraces += (ticks_finish - ticks_start);
		}

		/* ---------- update synapses ---------- */
		if (profiling) ticks_start = getticks();

		sk_mpi_updatesynapses(m->synapses, m->traces_syn,
								m->traces_neu, neuronoutputs,
								m->dn, dt, &m->p);

		if (profiling) {
			ticks_finish = getticks();
			updatingsynstrengths += (ticks_finish - ticks_start);
		}


		/* advance the buffer */
		if (profiling) ticks_start = getticks();

		dn_mpi_advance(m->dn);

		if (profiling) {
			ticks_finish = getticks();
			advancingbuffer += (ticks_finish - ticks_start);
			totalticks_finish = getticks();
			totaltickscum += (totalticks_finish - totalticks_start);
		}
	}


	/* -------------------- Performance Analysis -------------------- */
	if (profiling) {
		double cycletime, cumtime = 0.0;
		double firingrate;
		double *firingrates_g=0;
		double *gettinginputs_g=0;
		double *updatingsyntraces_g=0;
		double *updatingneurons_g=0;
		double *spikechecking_g=0;
		double *pushingoutput_g=0;
		double *updatingsynstrengths_g=0;
		double *advancingbuffer_g=0;
		double *totaltime_g=0;
		double *updatingneutraces_g=0;
		if (commrank==0) {
			firingrates_g = calloc(commsize, sizeof(double));
			gettinginputs_g = calloc(commsize, sizeof(double));
			updatingsyntraces_g = calloc(commsize, sizeof(double));
			updatingneurons_g = calloc(commsize, sizeof(double));
			spikechecking_g = calloc(commsize, sizeof(double));
			pushingoutput_g = calloc(commsize, sizeof(double));
			updatingsynstrengths_g = calloc(commsize, sizeof(double));
			updatingneutraces_g = calloc(commsize, sizeof(double));
			advancingbuffer_g = calloc(commsize, sizeof(double));
			totaltime_g = calloc(commsize, sizeof(double));
		}
		firingrate = ((double) numspikes) / (((double) n_l)*tp.dur);
		if (commrank==0) 
			MPI_Gather(&firingrate, 1, MPI_DOUBLE, firingrates_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&firingrate, 1, MPI_DOUBLE, NULL, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);

		cycletime = 1000.0*(((long double) gettinginputs) / 
					((long double) CLOCKSPEED))/numsteps;
		cumtime += cycletime;
		if (commrank==0) 
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, gettinginputs_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, NULL, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);

		cycletime = 1000.0*(((long double) updatingsyntraces) / 
					((long double) CLOCKSPEED))/numsteps;
		cumtime += cycletime;
		if (commrank==0) 
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, updatingsyntraces_g,
						1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, NULL, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);

		cycletime = 1000.0*(((long double)updatingneurons) / 
					((long double) CLOCKSPEED))/numsteps;
		cumtime += cycletime;
		if (commrank==0) 
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, updatingneurons_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE, 0,
						MPI_COMM_WORLD);

		cycletime = 1000.0*(((long double)spikechecking) / 
					((long double) CLOCKSPEED))/numsteps;
		cumtime += cycletime;
		if (commrank==0) 
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, spikechecking_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, NULL, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);

		cycletime = 1000.0*(((long double) pushingoutput) / 
					((long double) CLOCKSPEED)) / numsteps;
		cumtime += cycletime;
		if (commrank==0) 
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, pushingoutput_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, NULL, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);

		cycletime = 1000.0*(((long double) updatingneutraces) / 
					((long double) CLOCKSPEED))/numsteps;
		cumtime += cycletime;
		if (commrank==0) 
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, updatingneutraces_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, NULL, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);

		cycletime = 1000.0*(((long double) updatingsynstrengths) / 
					((long double) CLOCKSPEED))/numsteps;
		cumtime += cycletime;
		if (commrank==0) 
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, updatingsynstrengths_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, NULL, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);

		cycletime = 1000.0*(((long double) advancingbuffer) / 
					((long double) CLOCKSPEED))/numsteps;
		cumtime += cycletime;
		if (commrank==0) 
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, advancingbuffer_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&cycletime, 1, MPI_DOUBLE, NULL, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);

		double totaltimecum = 1000.0 * (((long double) totaltickscum) / 
								((long double) CLOCKSPEED))/numsteps;
		if (commrank==0) 
			MPI_Gather(&totaltimecum, 1, MPI_DOUBLE, totaltime_g, 1,
						MPI_DOUBLE, 0, MPI_COMM_WORLD);
		else
			MPI_Gather(&totaltimecum, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE,
						0, MPI_COMM_WORLD);

		if (commrank==0) {
			FILE *f;
			f = fopen(perfFileName, "a");
			fprintf(f, "\n----------------------------------------\n");
			fprintf(f, "Number of processes: %d\n", commsize);
			fprintf(f, "----------------------------------------\n");
			fprintf(f, "Sampling Frequency: %g\n", m->p.fs);
			fprintf(f, "Number of Neurons: %g\n", m->p.num_neurons);
			fprintf(f, "Probability of Contact: %g\n", m->p.p_contact);
			fprintf(f, "Maximum delay: %g\n", m->p.maxdelay);
			fprintf(f, "----------------------------------------\n");
			fprintf(f, "Firing rate: %g\n",  meantime(firingrates_g, commsize));
			fprintf(f, "----------------------------------------\n");
			fprintf(f, "Getting inputs:\t\t %f (ms)\n",
						meantime(gettinginputs_g, commsize));
			fprintf(f, "Update syntraces:\t %f (ms)\n",
						meantime(updatingsyntraces_g, commsize));
			fprintf(f, "Update neurons:\t\t %f (ms)\n",
						meantime(updatingneurons_g, commsize));
			fprintf(f, "Check spiked:\t\t %f (ms)\n",
						meantime(spikechecking_g, commsize));
			fprintf(f, "Pushing buffer:\t\t %f (ms)\n",
						meantime(pushingoutput_g, commsize));
			fprintf(f, "Updating neuron traces:\t %f (ms)\n",
						meantime(updatingneurons_g, commsize));
			fprintf(f, "Updating synapses:\t %f (ms)\n",
						meantime(updatingsynstrengths_g, commsize));
			fprintf(f, "Advancing buffer:\t %f (ms)\n",
						meantime(advancingbuffer_g, commsize));
			fprintf(f, "Total cycle time:\t %f (ms)\n",
						meantime(totaltime_g, commsize));
			fprintf(f, "\nTime per second: \t %f (ms)\n",
						maxtime(totaltime_g, commsize)*m->p.fs);
			fclose(f);
		}
		free(firingrates_g); 
		free(gettinginputs_g);
		free(updatingsyntraces_g);
		free(updatingneurons_g);
		free(spikechecking_g);
		free(pushingoutput_g);
		free(updatingsynstrengths_g);
		free(updatingneutraces_g);
		free(advancingbuffer_g);
		free(totaltime_g);
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

	/* Give each rank a different seed */
	srand(commrank+1);

	/* default neuron params (Izhikevich RS and FS) */
	FLOAT_T g_v_default = -65.0;
	FLOAT_T g_u_default = -13.0;

	FLOAT_T g_a_exc  = 0.02;
	FLOAT_T g_d_exc  = 8.0;

	FLOAT_T g_a_inh  = 0.1;
	FLOAT_T g_d_inh  = 2.0;

	/* set up delnet framework -- MAYBE BCAST THIS*/
	su_mpi_readmparameters(&m->p, mparamfilename);

	n = m->p.num_neurons;
	n_exc = (unsigned int) ((double) n * m->p.p_exc);
	size_t maxnode = dn_mpi_maxnode(commrank, commsize, n);
	size_t nodeoffset =  dn_mpi_nodeoffset(commrank, commsize, n);
	m->commrank = commrank;
	m->commsize = commsize;
	m->maxnode = maxnode;
	m->nodeoffset = nodeoffset;

	/* make sure all using the same graph -- CHANGE THIS TO BCAST! */
	if (commrank == 0) {
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

	for (i=0; i<maxnode; i++) {
		if (nodeoffset + i < n_exc)
			su_mpi_neuronset(&neurons[i], g_v_default, g_u_default, g_a_exc, g_d_exc);
		else
			su_mpi_neuronset(&neurons[i], g_v_default, g_u_default, g_a_inh, g_d_inh);
	}

	/* initialize synapse weights */
	if (DEBUG) printf("Initializing synapses on rank %d\n", commrank);
	unsigned int numsyn_exc = 0;

	// Find out number of excitatory synapses
	if (n_exc >= m->nodeoffset && n_exc < m->nodeoffset + m->maxnode) {
		if (n_exc < 1) { printf("Need at least one excitatory neuron!\n"); exit(-1); }
		numsyn_exc = m->dn->lineoffset_out + m->dn->nodes[(n_exc-1) - m->nodeoffset].idx_outbuf
						+ m->dn->nodes[(n_exc-1) - m->nodeoffset].num_in;
		for (int q=0; q < commsize; q++) {
			if (q != commrank) {
				MPI_Send(&numsyn_exc,
						sizeof(unsigned int),
						MPI_UNSIGNED, 
						q, 
						101,
						MPI_COMM_WORLD);
			}
		}
	} else {
		MPI_Recv(&numsyn_exc,
				sizeof(unsigned int),
				MPI_UNSIGNED,
				MPI_ANY_SOURCE,
				101,
				MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

	}

	if (DEBUG) {
		printf("On rank %d we think there are %d excitatory synapses.\n",
				commrank, numsyn_exc);
	}

	FLOAT_T *synapses_local = calloc(m->dn->numlinesin_l, sizeof(FLOAT_T));
	traces_syn = calloc(m->dn->numlinesin_l, sizeof(FLOAT_T));		

	//numsyn_exc = 83893; <- number in serial implementation
	unsigned int i_g;
	for (i=0; i< m->dn->numlinesin_l; i++) {
		i_g = i + m->dn->lineoffset_out;
		synapses_local[i] = m->dn->sourceidx_g[i_g] < numsyn_exc ? m->p.w_exc : m->p.w_inh;
	}
	
	m->numinputneurons = 100; 	// <- refactor out -- now in trial params
	m->neurons = neurons;
	m->traces_neu = traces_neu;
	m->traces_syn = traces_syn;
	m->synapses = synapses_local;

	if (DEBUG) printf("About to free graph on rank %d\n", commrank);
	free(graph);
	//free(synapses);

	return m;
}


/* loading and freeing models */

void su_mpi_savelocalmodel(su_mpi_model_l *m, FILE *f)
{
	/* Write data */	
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
}

/*
 * Parallelize properly later -- here essentially sequential. Need to calculate
 * offsets in advance, then can use MPI I/O.
 */
void su_mpi_globalsave(su_mpi_model_l *m_l, char *name, int commrank, int commsize)
{
	char filename[512];
	strcpy(filename, name);
	strcat(filename, "_model.bin");

	FILE *f = 0; 

	if (commrank == 0) {
		f = fopen(filename, "wb");
		fwrite(&commsize, sizeof(int), 1, f);
		fclose(f);
	}

	if (commsize == 1) {
		f = fopen(filename, "ab");
		su_mpi_savelocalmodel(m_l, f);
		fclose(f);
	} else {
		if (commrank == 0) {
			int msg = 1;
			f = fopen(filename, "ab");
			su_mpi_savelocalmodel(m_l, f);
			fclose(f);
			MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		} else if (commrank < commsize-1) {
			int msg;
			MPI_Recv(&msg, 1, MPI_INT, commrank-1, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE);
			f = fopen(filename, "ab");
			su_mpi_savelocalmodel(m_l, f);
			fclose(f);
			MPI_Send(&msg, 1, MPI_INT, commrank+1, 0, MPI_COMM_WORLD);
		} else {
			int msg;
			MPI_Recv(&msg, 1, MPI_INT, commrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			f = fopen(filename, "ab");
			su_mpi_savelocalmodel(m_l, f);
			fclose(f);
		}
	}
}


su_mpi_model_l *su_mpi_loadlocalmodel(FILE *f)
{

	su_mpi_model_l *m = malloc(sizeof(su_mpi_model_l));
	size_t loadsize;

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


su_mpi_model_l *su_mpi_globalload(char *name, int commrank, int commsize)
{

	su_mpi_model_l *m_l; 
	size_t loadsize;
	int readcommsize;

	char filename[512];
	strcpy(filename, name);
	strcat(filename, "_model.bin");

	FILE *f = fopen(filename, "rb");

	if (commrank == 0) {
		loadsize = fread(&readcommsize, sizeof(int), 1, f);
		if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); };
		if (readcommsize != commsize) {
			printf("Model uses %d processes, but program launched with %d processes. Exiting.\n",
					readcommsize, commsize);
			fclose(f);
			exit(-1);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (commsize == 1) {
		m_l = su_mpi_loadlocalmodel(f);
	} else {
		if (commrank == 0) {
			int msg = 1;
			m_l = su_mpi_loadlocalmodel(f);
			if (commrank != m_l->dn->commrank || commsize != m_l->dn->commsize) {
				printf("MPI Size or Rank mismatch while loading data.\n");
				exit(-1);
			}
			MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		} else if (commrank < commsize-1) {
			int msg;
			MPI_Recv(&msg, 1, MPI_INT, commrank-1, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE);
			m_l = su_mpi_loadlocalmodel(f);
			if (commrank != m_l->dn->commrank || commsize != m_l->dn->commsize) {
				printf("MPI Size or Rank mismatch while loading data.\n");
				exit(-1);
			}
			MPI_Send(&msg, 1, MPI_INT, commrank+1, 0, MPI_COMM_WORLD);
		} else {
			int msg;
			MPI_Recv(&msg, 1, MPI_INT, commrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			m_l = su_mpi_loadlocalmodel(f);
			if (commrank != m_l->dn->commrank || commsize != m_l->dn->commsize) {
				printf("MPI Size or Rank mismatch while loading data.\n");
				exit(-1);
			}
		}
	}

	fclose(f);

	return m_l;
}

void su_mpi_savemodel_l(su_mpi_model_l *m, char *name,
						int commsize, int commrank)
{
	char filename[512];
	char rankstr[64];

	/* Make file name */
	sprintf(rankstr, "_%d_%d_model.bin", commsize, commrank);
	strcpy(filename, name);
	strcat(filename, rankstr);

	/* Open file */
	FILE *f = fopen(filename, "wb");

	/* Write data */	
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

	/* Close file */
	fclose(f);
}


su_mpi_model_l *su_mpi_loadmodel_l(char *name, int commrank, int commsize)
{
	char filename[512];
	char rankstr[64];
	su_mpi_model_l *m = malloc(sizeof(su_mpi_model_l));
	size_t loadsize;

	/* Prepare file name */
	sprintf(rankstr, "_%d_%d_model.bin", commsize, commrank);
	strcpy(filename, name);
	strcat(filename, rankstr);

	printf("On rank %d loading file: %s\n", commrank, filename);

	FILE *f = fopen(filename, "rb");
	f = fopen(filename, "r");
	if (f == NULL) {
		perror(name);
		printf("Failed here!\n");
		exit(EXIT_FAILURE);
	}

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

	/* sanity check */
	if (commrank != m->dn->commrank || commsize != m->dn->commsize) {
		printf("MPI Size or Rank mismatch while loading data.\n");
		exit(-1);
	}

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

