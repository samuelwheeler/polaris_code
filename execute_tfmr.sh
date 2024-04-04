#!/bin/bash -l
#PBS -l walltime=1:00:00
#PBS -l select=2
#PBS -N tfmr_small_c4
#PBS -k doe
#PBS -j oe
#PBS -A tpc
#PBS -l filesystems=home:eagle
#PBS -M swwheeler@uchicago.edu                                                  
#PBS -m bae 
#PBS -q debug

cd $PBS_O_WORKDIR


module load conda/2023-10-04; conda activate base; module load cudatoolkit-standalone/11.4.4

# Count number of nodes assigned
NNODES=`wc -l < $PBS_NODEFILE`
# set 1 MPI rank per GPU
NRANKS_PER_NODE=4
# calculate total ranks
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} python multinode_big_dataset_streaming.py --save_every=10 --batch_size=8 --wandb_project='c4_pretrain' --buffer_size=1000 --wandb_name='tfmr_960_c4_proxy_comp' >& tfmr_960.out