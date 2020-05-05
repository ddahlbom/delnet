#ifndef PARAMUTILS_H
#define PARAMUTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_PARAM_NAME_LEN 100
#define MAX_LINE 200

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

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

EXTERNC paramlist *pl_init();
EXTERNC void pl_addparam(paramlist *pl, char *name, double val);
EXTERNC paramnode *pl_popparam(paramlist *pl);
EXTERNC int pl_lookup(paramlist *pl, char *name, double *outval);
EXTERNC void pl_free(paramlist *pl);
EXTERNC paramlist* pl_readparams(char * filename);
EXTERNC double pl_getvalue(paramlist *pl, char *filename);


#endif
