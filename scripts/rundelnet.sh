#!/bin/bash -x

if [ "x$SLURM_NPROCS" = "x" ]
then
  if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
  then
    SLURM_NTASKS_PER_NODE=1
  fi
  SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
else
  if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
  then
    SLURM_NTASKS_PER_NODE=`expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES`
  fi
fi

srun hostname -s | sort -u > /tmp/hosts.$SLURM_JOB_ID
awk "{ print \$0 \"-ib slots=$SLURM_NTASKS_PER_NODE\"; }" /tmp/hosts.$SLURM_JOB_ID >/tmp/tmp.$SLURM_JOB_ID
mv /tmp/tmp.$SLURM_JOB_ID /tmp/hosts.$SLURM_JOB_ID

module load spectrum-mpi
mpirun --bind-to core --report-bindings -hostfile /tmp/hosts.$SLURM_JOB_ID -np $SLURM_NPROCS /gpfs/u/home/CDAP/CDAPdhlb/barn/delnet/runtrial-mpi 0 /gpfs/u/home/CDAP/CDAPdhlb/barn/delnet/staging/PGTesterStager /gpfs/u/home/CDAP/CDAPdhlb/barn/delnet/staging/PGTesterStager

rm /tmp/hosts.$SLURM_JOB_ID

