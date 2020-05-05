#ifndef CUALLOCATE_H
#define CUALLOCATE_H

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif



void cuAlloc(void **block, size_t numelements, size_t typesize, int commRank);
void cuFree(void *block);
void cuAllocDouble(double **block, size_t numelements, int commRank);
void cuFreeDouble(double *block);

#endif
