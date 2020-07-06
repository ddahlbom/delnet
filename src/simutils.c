#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

//#include "delnet.h"
#include "delnetfixed.h"
#include "simutils.h"
#include "simkernels.h"
#include "paramutils.h"
#include "spkrcd.h"

#ifdef __amd64__
//#define CLOCKSPEED 1512000000
//#define CLOCKSPEED 2600000000
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

#define CLOCKSPEED 512000000
typedef unsigned long long ticks;
char perfFileName[] = "/gpfs/u/home/CDAP/CDAPdhlb/barn/delnet/performance_data.txt";
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


#define SU_DEBUG 0



/*
   TO DO
   - [ ] Synapse sorting (probably need to coordinate with delnet mods)
   - [ ] Make synapse weights loadable
   - [ ] Make multiple model types, increasing modularity (e.g. synapse struct)
*/





/*************************************************************
 *  Functions
 *************************************************************/
/* ----- Local Helper Functions ----- */
void checkfileload(FILE *f, char *name)
{
	if (f == NULL) {
		perror(name);
		printf("Failed here!\n");
		exit(EXIT_FAILURE);
	}
}


/* -------------------- Parameter Setting -------------------- */
void su_mpi_readmparameters(su_mpi_modelparams *p, char *name)
{
	char filename[MAX_NAME_LEN];
	strcpy(filename, name);
	strcat(filename, "_mparams.txt");
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

	p->a_exc = pl_getvalue(pl, "a_exc");
	p->d_exc = pl_getvalue(pl, "d_exc");
	p->a_inh = pl_getvalue(pl, "a_inh");
	p->d_inh = pl_getvalue(pl, "d_inh");
	p->v_default = pl_getvalue(pl, "v_default");
	p->u_default = pl_getvalue(pl, "u_default");

	pl_free(pl);
}


void su_mpi_readtparameters(su_mpi_trialparams *p, char *name)
{
	char filename[512];
	strcpy(filename, name);
	strcat(filename, "_tparams.txt");
	paramlist *pl = pl_readparams(filename);
	
	p->dur = pl_getvalue(pl, "dur");
	p->lambda = pl_getvalue(pl, "lambda");
	p->randspikesize = pl_getvalue(pl, "randspikesize");
	p->randinput = (bool) pl_getvalue(pl, "randinput");
	p->inhibition = (bool) pl_getvalue(pl, "inhibition");
	p->inputmode = (idx_t) pl_getvalue(pl, "inputmode");
	p->multiinputmode = (idx_t)  pl_getvalue(pl, "multiinputmode");
	p->inputweight = pl_getvalue(pl, "inputweight");
	p->recordstart = pl_getvalue(pl, "recordstart");
	p->recordstop = pl_getvalue(pl, "recordstop");
	p->lambdainput = pl_getvalue(pl, "lambdainput");
	p->inputrefractorytime = pl_getvalue(pl, "inputrefractorytime");

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
void su_mpi_neuronset(su_mpi_neuron *n, FLOAT_T a, FLOAT_T b, FLOAT_T c, FLOAT_T d)
{
	n->a = a;
	n->b = b;
	n->c = c;
	n->d = d;

	n->v = c;
	n->u = b*c;
}

/* -------------------- Graph Generation -------------------- */

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
							su_mpi_input *inputs, idx_t numinputs,
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
	idx_t n_l = m->dn->numnodes;
	FLOAT_T dt = 1.0/m->p.fs;
	IDX_T numsteps = tp.dur/dt;

	/* local state for simulation */
	FLOAT_T *neuroninputs, *neuronoutputs;
	FLOAT_T *nextrand = malloc(sizeof(FLOAT_T)*n_l);
	idx_t *neuronevents;
	idx_t numevents = 0;
	//FLOAT_T nextinputtime = 0.0;
	//bool waiting = true;
	neuroninputs = calloc(n_l, sizeof(FLOAT_T));
	neuronoutputs = calloc(n_l, sizeof(FLOAT_T));
	neuronevents = calloc(n_l, sizeof(idx_t));
	unsigned long int numspikes = 0, numrandspikes = 0;
	FLOAT_T t;

	/* initiate input */
	idx_t input_idx = 0;
	idx_t inputlen = 0;
	su_mpi_spike *input = 0;
	bool neednewinput = false;
	FILE *inputtimesfile = 0;
	char filename[MAX_NAME_LEN];
	double t_max_l = 0.0;

	if (tp.multiinputmode == MULTI_INPUT_MODE_RANDOM)
		input_idx = getrandom(numinputs);
	input = inputs[input_idx].spikes;
	inputlen = inputs[input_idx].len;
	sprintf(filename, "%s_instarttimes.txt", trialname);
	for (int i=0; i<inputlen; i++)
		if (input[i].t > t_max_l) t_max_l = input[i].t; // find last spike time on local rank

	/* find maximum accross ranks for coordinating input times */
	double *t_maxs = malloc(sizeof(double)*commsize);
	MPI_Allgather(&t_max_l, 1, MPI_DOUBLE, t_maxs, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	double t_max = 0.0;
	for (idx_t i=0; i<commsize; i++) 
		if (t_maxs[i] > t_max) t_max = t_maxs[i];
	
	if (commrank == 0)
		inputtimesfile = fopen(filename, "w");

	/* initialize random input states */
	for(size_t i=0; i<n_l; i++) nextrand[i] = sk_mpi_expsampl(tp.lambda);

	/* main simulation loop */
	for (size_t i=0; i<numsteps; i++) {
		if (profiling) totalticks_start = getticks(); 
		numevents = 0;

		/* ---------- calculate time update ---------- */
		t = dt*i;
		if (i%1000 == 0 && commrank == 0)
			printf("Time: %f\n", t);


		/* ---------- inputs ---------- */
		if (profiling) ticks_start = getticks();

		/* get inputs from delay net */
		sk_mpi_getinputs(neuroninputs, m->dn, m->synapses);

		/* put in forced input -- make this a function in kernels! */
		neednewinput = sk_mpi_forcedinput(m, input, inputlen, input_idx, neuroninputs, t, dt, t_max,
										  &tp, commrank, commsize, inputtimesfile, nextrand ); 
		if (neednewinput) {
			if (commrank == 0) {
				if (tp.multiinputmode == MULTI_INPUT_MODE_SEQUENTIAL) {
					input_idx = (input_idx + 1) % numinputs;
				}
				else if (tp.multiinputmode == MULTI_INPUT_MODE_RANDOM) {
					input_idx = getrandom(numinputs);
				}
				MPI_Bcast(&input_idx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
			} else {
				MPI_Bcast(&input_idx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
			}

			input = inputs[input_idx].spikes;
			inputlen = inputs[input_idx].len;
		}

		/* ---------- put in random noise ---------- */

		numrandspikes += sk_mpi_poisnoise(neuroninputs, nextrand, t, n_l, &tp);


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

		numevents = sk_mpi_checkspiking(m->neurons, neuronoutputs,
										neuronevents, n_l, t,
										sr, m->dn->nodeoffsetglobal,
										tp.recordstart, tp.recordstop);
		numspikes += numevents;

		if (profiling) {
			ticks_finish = getticks();
			spikechecking += (ticks_finish - ticks_start);
		}

		/* ---------- push the neuron output into the buffer ---------- */
		if (profiling) ticks_start = getticks(); 

		//for (size_t k=0; k<n_l; k++)
		dnf_pushevents(m->dn, neuronevents, numevents, commrank, commsize);

		if (profiling) {
			ticks_finish = getticks();
			pushingoutput += (ticks_finish - ticks_start);
		}


		/* ---------- update synapse traces ---------- */
		if (profiling) ticks_start = getticks();

		sk_mpi_updatesynapsetraces(m->traces_syn, m->dn->nodeinputbuf, m->dn, dt, &m->p);

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

		dnf_advance(m->dn);

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
			checkfileload(f, perfFileName);
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


unsigned long *su_mpi_loadgraph(char *name, su_mpi_model_l *m, int commrank)
{
	long long *graph=0, n;
	unsigned long *ugraph = 0;
	char filename[MAX_NAME_LEN];
	strcpy(filename, name);
	strcat(filename, "_graph.bin");

	if (commrank == 0) {
		FILE *f;
		size_t loadsize;

		f = fopen(filename, "rb");

		checkfileload(f, filename);
		loadsize = fread(&n, sizeof(long int), 1, f);
		if (loadsize != 1) { printf("Failed to load graph.\n"); exit(-1); }
		if (n != m->p.num_neurons) {
			printf("Graph size doesn't match parameter file!\n");
			exit(-1);
		}
		graph = malloc(sizeof(long int)*n*n);
		loadsize = fread(graph, sizeof(long int), n*n, f);
		if (loadsize != n*n) { printf("Failed to load graph.\n"); exit(-1); }

		fclose(f);

		ugraph = malloc(sizeof(unsigned long)*n*n);
		for (size_t i=0; i<n*n; i++)
			ugraph[i] = (unsigned long) graph[i];
		free(graph);


		if (MPI_SUCCESS != MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD)) {
			printf("MPI Broadcast failure (graph loading).\n");
			exit(-1);
		}
		if (MPI_SUCCESS != MPI_Bcast(ugraph, n*n, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD)) {
			printf("MPI Broadcast failure (graph loading).\n");
			exit(-1);
		}

	} else {
		if (MPI_SUCCESS != MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD)) {
			printf("MPI Broadcast failure (graph loading).\n");
			exit(-1);
		}
		ugraph = malloc(sizeof(unsigned long)*n*n);
		if (MPI_SUCCESS != MPI_Bcast(ugraph, n*n, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD)) {
			printf("MPI Broadcast failure (graph loading).\n");
			exit(-1);
		}
	}

	return ugraph;
}

MPI_Datatype commitmpineurontype() 
{
	const int nitems = 6;
	int blocklengths[6] = { 1, 1, 1, 1, 1, 1 };
	MPI_Datatype types[6] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
							  MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	MPI_Datatype mpi_neuron_type;
	MPI_Aint offsets[6];

	offsets[0] = offsetof(su_mpi_neuron, v);
	offsets[1] = offsetof(su_mpi_neuron, u);
	offsets[2] = offsetof(su_mpi_neuron, a);
	offsets[3] = offsetof(su_mpi_neuron, b);
	offsets[4] = offsetof(su_mpi_neuron, c);
	offsets[5] = offsetof(su_mpi_neuron, d);

	MPI_Type_create_struct(nitems, blocklengths, offsets,
						   types, &mpi_neuron_type);
	if (MPI_SUCCESS != MPI_Type_commit(&mpi_neuron_type)) {
		printf("Failed to commit custom MPI neuron type!\n");
		exit(-1);
	}

	return mpi_neuron_type;
}

/* TO-DO: Proper MPI IO implementation -- this isn't parallelized */
su_mpi_neuron *su_mpi_loadlocalneurons(char *name, su_mpi_model_l *m)
{
	FILE *f;
	char filename[MAX_NAME_LEN];
	size_t loadsize;
	long int numneurons;
	MPI_Datatype mpi_neuron_type = commitmpineurontype();
	su_mpi_neuron *neurons_g = 0; 
	su_mpi_neuron *neurons_l = malloc(sizeof(su_mpi_neuron)*m->maxnode);

	if (m->commrank == 0) {
		strcpy(filename, name);
		strcat(filename, "_neurons.bin");
		f = fopen(filename, "rb");
		checkfileload(f, filename);

		/* Load number of neurons and check */
		loadsize = fread(&numneurons, sizeof(long int), 1, f);
		if (loadsize != 1) {
			printf("Failed to load number of neurons\n");
			exit(-1);
		}
		if (m->p.num_neurons != numneurons) {
			printf("Bad neuron data: %f (parameters), %ld (file)\n",
					m->p.num_neurons, numneurons);
			exit(-1);
		}

		/* Load neuron data */
		neurons_g = malloc(sizeof(su_mpi_neuron)*numneurons);
		double a,b,c,d;
		for (int i=0; i<numneurons; i++) {
			loadsize = fread(&a, sizeof(double), 1, f);
			if (loadsize != 1) {
				printf("Failed to load neuron %d\n", i); exit(-1);
			}
			loadsize = fread(&b, sizeof(double), 1, f);
			if (loadsize != 1) {
				printf("Failed to load neuron %d\n", i); exit(-1);
			}
			loadsize = fread(&c, sizeof(double), 1, f);
			if (loadsize != 1) {
				printf("Failed to load neuron %d\n", i); exit(-1);
			}
			loadsize = fread(&d, sizeof(double), 1, f);
			if (loadsize != 1) {
				printf("Failed to load neuron %d\n", i); exit(-1);
			}
			su_mpi_neuronset(&neurons_g[i], a, b, c, d);
		}
		fclose(f);

		/* Data for Scatterv */
		int *lens = malloc(sizeof(int)*m->commsize);
		int *displs = malloc(sizeof(int)*m->commsize);
		for (int i=0; i<m->commsize; i++) {
			lens[i] = dnf_maxnode(i, m->commsize, numneurons);
			displs[i] = dnf_nodeoffset(i, m->commsize, numneurons);
		}

		/* Scatter data */
		MPI_Scatterv(neurons_g, lens, displs, mpi_neuron_type,
					 neurons_l, m->maxnode, mpi_neuron_type,
					 0, MPI_COMM_WORLD);

		/* Clean up */
		free(neurons_g);
		free(lens);
		free(displs);
	}
	else {
		neurons_l = malloc(sizeof(su_mpi_neuron)*m->maxnode);
		MPI_Scatterv(NULL, NULL, NULL, NULL,
					 neurons_l, m->maxnode, mpi_neuron_type,
					 0, MPI_COMM_WORLD);
	}
	return neurons_l;
}


/* make models */
su_mpi_model_l *su_mpi_izhimodelfromgraph(char *name, int commrank, int commsize)
{
	unsigned int n, n_exc, i;
	unsigned long *graph = 0;
	su_mpi_model_l *m = malloc(sizeof(su_mpi_model_l));

	/* Give each rank a different seed */
	srand(commrank+1);

	/* set up delnet framework -- MAYBE BCAST THIS*/
	su_mpi_readmparameters(&m->p, name);

	n = m->p.num_neurons;
	n_exc = (unsigned int) ((double) n * m->p.p_exc);
	size_t maxnode = dnf_maxnode(commrank, commsize, n);
	size_t nodeoffset =  dnf_nodeoffset(commrank, commsize, n);
	m->commrank = commrank;
	m->commsize = commsize;
	m->maxnode = maxnode;
	m->nodeoffset = nodeoffset;

	/* load graph on all ranks (uses bcast) */ 
	graph = su_mpi_loadgraph(name, m, commrank);
	m->dn = dnf_delaynetfromgraph(graph, n, commrank, commsize);
	free(graph);


	/* load neuron parameter information */
	su_mpi_neuron *neurons  = 0;
	neurons = su_mpi_loadlocalneurons(name, m);

	/* set up other state for simulation */
	FLOAT_T *traces_neu 	= calloc(maxnode, sizeof(FLOAT_T));
	FLOAT_T *traces_syn; 	

	/* set up synapses */
	FLOAT_T *synapses_local = calloc(m->dn->numbufferstotal, sizeof(FLOAT_T));
	traces_syn = calloc(m->dn->numbufferstotal, sizeof(FLOAT_T));		

	//unsigned int i_g;
	idx_t numneuronexcitatory = m->p.p_exc * m->p.num_neurons;
	for (i=0; i< m->dn->numbufferstotal; i++) {
		//i_g = i + m->dn->bufferoffsetglobal;
		//synapses_local[i] = m->dn->sourceidx_g[i_g] < numsyn_exc ? m->p.w_exc : m->p.w_inh;
		synapses_local[i] =
			m->dn->buffersourcenodes[i] <= numneuronexcitatory ?
			m->p.w_exc :
			m->p.w_inh;
	}
	
	m->neurons = neurons;
	m->traces_neu = traces_neu;
	m->traces_syn = traces_syn;
	m->synapses = synapses_local;

	//free(synapses);
	printf("On rank %d: %g\n", commrank, m->p.num_neurons);

	return m;
}

/* --------------- loading and freeing models --------------- */



void checksizeandrank(su_mpi_model_l *m, int commrank, int commsize)
{
	if (commrank != m->dn->commrank || commsize != m->dn->commsize) {
		printf("MPI Size or Rank mismatch while loading data.\n");
		printf("Current size: %d, Loaded size: %d\n", commsize, m->dn->commsize);
		printf("Current rank: %d, Loaded rank: %d\n", commrank, m->dn->commrank);
		exit(-1);
	}
}


/* ----- Save synapse weights ----- */


/*
 * This one could be easily parallelized with MPI read and write...
 */
void su_mpi_savesynapses(su_mpi_model_l *m, char *name,
						 int commrank, int commsize)
{
	FILE *f;
	char filename[512];
	IDX_T *synlens = 0;
	int *synlens_i = 0;
	int *offsets = 0;
	FLOAT_T *synapses_g = 0;
	idx_t *sourcenodes_g = 0;
	idx_t *destnodes_g = 0;
	unsigned short *delays_g = 0;
	unsigned short *delays_l = 0;
	unsigned long totallen = 0;


	strcpy(filename, name);
	strcat(filename, "_synapses.bin");

	/* Write length at head for parsing */
	if (commrank == 0) synlens = malloc(sizeof(IDX_T)*commsize);

	MPI_Gather(&m->dn->numbufferstotal, 1, mpi_idx_t,
			   synlens, 1, mpi_idx_t, 0,
			   MPI_COMM_WORLD);

	if (commrank == 0) {
		for (int i=0; i<commsize; i++) totallen += synlens[i];
		synapses_g = malloc(sizeof(FLOAT_T)*totallen);
		sourcenodes_g = malloc(sizeof(idx_t)*totallen);
		destnodes_g = malloc(sizeof(idx_t)*totallen);
		delays_g = malloc(sizeof(unsigned short)*totallen);
		synlens_i = malloc(sizeof(IDX_T)*commsize);

		f = fopen(filename, "wb");
		checkfileload(f, filename);
		fwrite(&totallen, sizeof(unsigned long), 1, f);	
		fclose(f);

		for (int i=0; i<commsize; i++) synlens_i[i] = (int) synlens[i];
		offsets = len_to_offsets(synlens_i, commsize);
	}

	delays_l = malloc(sizeof(unsigned short)*m->dn->numbufferstotal);
	for (idx_t i=0; i<m->dn->numbufferstotal; i++)
		delays_l[i] = m->dn->buffers[i].delaylen;

	MPI_Gatherv(m->synapses, m->dn->numbufferstotal, MPI_DOUBLE,
				synapses_g, synlens_i, offsets, MPI_DOUBLE, 0,
				MPI_COMM_WORLD);
	MPI_Gatherv(m->dn->buffersourcenodes, m->dn->numbufferstotal,
				MPI_UNSIGNED_LONG, sourcenodes_g, synlens_i, offsets,
				MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	MPI_Gatherv(m->dn->bufferdestnodes, m->dn->numbufferstotal,
				MPI_UNSIGNED_LONG, destnodes_g, synlens_i, offsets,
				MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	MPI_Gatherv(delays_l, m->dn->numbufferstotal,
				MPI_UNSIGNED_SHORT, delays_g, synlens_i, offsets,
				MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);

	if (commrank == 0) {
		//synapses_sorted = malloc(sizeof(FLOAT_T)*totallen);
		// FIGURE THIS ONE OUT LATER!!!! UNSORTED NOW
		//synapses_sorted = synapses_g;
		//for (int i=0; i<totallen; i++) 
		//	synapses_sorted[i] = synapses_g[ m->dn->destidx_g[i] ];

		f = fopen(filename, "ab");
		checkfileload(f, filename);
		fwrite(synapses_g, sizeof(FLOAT_T), totallen, f);	
		fwrite(sourcenodes_g, sizeof(idx_t), totallen, f);	
		fwrite(destnodes_g, sizeof(idx_t), totallen, f);	
		fwrite(delays_g, sizeof(unsigned short), totallen, f);	
		fclose(f);

		free(offsets);
		free(synlens);
		free(synlens_i);
		free(synapses_g);
		free(sourcenodes_g);
		free(destnodes_g);
		free(delays_l);
		free(delays_g);
		//free(synapses_sorted);
	}
}


/* ----- Rank-local Save and Load ----- */
void su_mpi_savelocalmodel(su_mpi_model_l *m, FILE *f)
{
	/* Write data */	
	dnf_save(m->dn, f);

	fwrite(&m->commrank, sizeof(int), 1, f);
	fwrite(&m->commsize, sizeof(int), 1, f);
	fwrite(&m->maxnode, sizeof(size_t), 1, f);
	fwrite(&m->nodeoffset, sizeof(size_t), 1, f);
	//fwrite(&m->numsyn, sizeof(IDX_T), 1, f);
	fwrite(&m->p, sizeof(su_mpi_modelparams), 1, f);
	fwrite(m->neurons, sizeof(su_mpi_neuron), m->dn->numnodes, f);
	fwrite(m->traces_neu, sizeof(FLOAT_T), m->dn->numnodes, f);
	fwrite(m->traces_syn, sizeof(FLOAT_T), m->dn->numbufferstotal, f);
	fwrite(m->synapses, sizeof(FLOAT_T), m->dn->numbufferstotal, f);
}


su_mpi_model_l *su_mpi_loadlocalmodel(FILE *f)
{
	su_mpi_model_l *m = malloc(sizeof(su_mpi_model_l));
	size_t loadsize;

	m->dn = dnf_load(f);

	loadsize = fread(&m->commrank, sizeof(int), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->commsize, sizeof(int), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->maxnode, sizeof(size_t), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->nodeoffset, sizeof(size_t), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	//loadsize = fread(&m->numsyn, sizeof(IDX_T), 1, f);
	//if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	loadsize = fread(&m->p, sizeof(su_mpi_modelparams), 1, f);
	if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); }

	m->neurons = malloc(sizeof(su_mpi_neuron)*m->dn->numnodes);
	loadsize = fread(m->neurons, sizeof(su_mpi_neuron), m->dn->numnodes, f);
	if (loadsize != m->dn->numnodes) { printf("Failed to load model.\n"); exit(-1); }

	m->traces_neu = malloc(sizeof(FLOAT_T)*m->dn->numnodes);
	loadsize = fread(m->traces_neu, sizeof(FLOAT_T), m->dn->numnodes, f);
	if (loadsize != m->dn->numnodes) { printf("Failed to load model.\n"); exit(-1); }

	m->traces_syn = malloc(sizeof(FLOAT_T)*m->dn->numbufferstotal);
	loadsize = fread(m->traces_syn, sizeof(FLOAT_T), m->dn->numbufferstotal, f);
	if (loadsize != m->dn->numbufferstotal) { printf("Failed to load model.\n"); exit(-1); }

	m->synapses = malloc(sizeof(FLOAT_T)*m->dn->numbufferstotal);
	loadsize = fread(m->synapses, sizeof(FLOAT_T), m->dn->numbufferstotal, f);
	if (loadsize != m->dn->numbufferstotal) { printf("Failed to load model.\n"); exit(-1); }

	return m;
}


/* ----- Unified (Global) Save and Load ----- */
void su_mpi_globalsave(su_mpi_model_l *m_l, char *name, int commrank, int commsize)
{
	FILE *f = 0; 
	char filename[512];

	strcpy(filename, name);
	strcat(filename, "_model.bin");

	/* Write the the number of processes at head */
	if (commrank == 0) {
		f = fopen(filename, "wb");
		fwrite(&commsize, sizeof(int), 1, f);
		fclose(f);
	}

	/* Now write the data in rank order */
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
			MPI_Recv(&msg, 1, MPI_INT, commrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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


su_mpi_model_l *su_mpi_globalload(char *name, int commrank, int commsize)
{
	FILE *f; 
	su_mpi_model_l *m_l; 
	size_t loadsize;
	int readcommsize;
	long int position = 0;
	char filename[512];

	strcpy(filename, name);
	strcat(filename, "_model.bin");

	/* Read the number of processes at the head and check if matches commsize */
	if (commrank == 0) {
		f = fopen(filename, "rb");
		checkfileload(f, filename);
		loadsize = fread(&readcommsize, sizeof(int), 1, f);
		if (loadsize != 1) { printf("Failed to load model.\n"); exit(-1); };
		if (readcommsize != commsize) {
			printf("Model uses %d processes, but program launched with %d processes. Exiting.\n",
					readcommsize, commsize);
			fclose(f);
			exit(-1);
		}
		position = ftell(f);
		fclose(f);
	}

	/* Read out the rest of the data in rank order */
	if (commsize == 1) {
		f = fopen(filename, "rb");
		checkfileload(f, filename);
		fseek(f, position, SEEK_SET);
		m_l = su_mpi_loadlocalmodel(f);
		fclose(f);
	} else {
		if (commrank == 0) {
			f = fopen(filename, "rb");
			checkfileload(f, filename);
			fseek(f, position, SEEK_SET);
			m_l = su_mpi_loadlocalmodel(f);
			checksizeandrank(m_l, commrank, commsize);
			position = ftell(f);
			fclose(f);
			MPI_Send(&position, 1, MPI_LONG, 1, 0, MPI_COMM_WORLD);
		} else if (commrank < commsize-1) {
			MPI_Recv(&position, 1, MPI_LONG, commrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			f = fopen(filename, "rb");
			checkfileload(f, filename);
			fseek(f, position, SEEK_SET);
			m_l = su_mpi_loadlocalmodel(f);
			checksizeandrank(m_l, commrank, commsize);
			position = ftell(f);
			fclose(f);
			MPI_Send(&position, 1, MPI_LONG, commrank+1, 0, MPI_COMM_WORLD);
		} else {
			MPI_Recv(&position, 1, MPI_LONG, commrank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			f = fopen(filename, "rb");
			checkfileload(f, filename);
			fseek(f, position, SEEK_SET);
			m_l = su_mpi_loadlocalmodel(f);
			checksizeandrank(m_l, commrank, commsize);
			position = ftell(f);
			fclose(f);
		}
	}
	return m_l;
}


/* --------------- Free Model Memory -------------------- */
void su_mpi_freemodel_l(su_mpi_model_l *m) {
	dnf_freedelaynet(m->dn);
	free(m->neurons);
	free(m->traces_neu);
	free(m->traces_syn);
	free(m->synapses);
	free(m);
}
