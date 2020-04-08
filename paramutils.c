#include "paramutils.h"

paramlist *pl_init()
{
	paramlist *pl = malloc(sizeof(paramlist)); 
	pl->count=0;
	pl->head=NULL;
	return pl;
}


void pl_addparam(paramlist *pl, char *name, double val)
{
	paramnode *node = malloc(sizeof(paramnode));
	node->value = val;
	strcpy(node->name, name);
	node->next = pl->head;
	pl->head = node;
}


paramnode *pl_popparam(paramlist *pl)
{
	if (pl->head == NULL) return NULL;
	paramnode *temp;
	temp = pl->head;
	pl->head = pl->head->next;
	return temp;
}


int pl_lookup(paramlist *pl, char *name, double *outval)
{
	unsigned char found = 0;
	paramnode *current;
	current = pl->head;
	while ( (current != NULL) && !found ) {
		if (strcmp(current->name, name) == 0) {
			*outval = current->value;
			return 0;
		}
		current = current->next;
	}
	return -1;
}


void pl_free(paramlist *pl)
{
	paramnode *node;
	while ( (node = pl_popparam(pl)) != NULL ) free(node);
	free(pl);
}

paramlist* pl_readparams(char * filename)
{
	FILE *paramf;
	double val;
	char line[MAX_LINE], name[MAX_PARAM_NAME_LEN];
	paramlist *pl = pl_init();

	if ( (paramf = fopen(filename, "r")) == NULL ) {
		printf("No file with name %s\n", filename);
		exit(-1);
	}

	while ( fgets(line, MAX_LINE, paramf) != NULL) {
		sscanf(line, "%s %lf", name, &val);
		pl_addparam(pl, name, val);
	}
	fclose(paramf);

	return pl;
}

double pl_getvalue(paramlist *pl, char *pname) 
{
	double val;
	if ( pl_lookup(pl, pname, &val) != 0) {
		printf("\"%s\" is not in the parameter list! Exiting...\n", pname);
		exit(-1);
	}
	return val;
}
