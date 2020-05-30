#ifndef PARAMUTILS_H
#define PARAMUTILS_H


#define MAX_PARAM_NAME_LEN 100
#define MAX_LINE 200


/* -------------------- Structures -------------------- */
typedef struct paramnode_s {
	char name[MAX_PARAM_NAME_LEN];
	double value;	
	struct paramnode_s *next;
} paramnode;

typedef struct paramlist_s {
	unsigned int count;
	paramnode *head;
} paramlist;


/* -------------------- Function Declarations -------------------- */

paramlist *pl_init();
void pl_addparam(paramlist *pl, char *name, double val);
paramnode *pl_popparam(paramlist *pl);
int pl_lookup(paramlist *pl, char *name, double *outval);
void pl_free(paramlist *pl);
paramlist* pl_readparams(char * filename);
double pl_getvalue(paramlist *pl, char *filename);


#endif
