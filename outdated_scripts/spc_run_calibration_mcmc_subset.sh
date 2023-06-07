#!/bin/sh
#SBATCH --partition=t1small
#SBATCH --ntasks=48
#SBATCH --tasks-per-node=24

# activate environment
module load lang/Anaconda3/5.3.0
source activate oggm_env_v02

# split glaciers into batches for different nodes
python spc_split_glaciers_random.py -n_batches=$SLURM_JOB_NUM_NODES

# list rgi_glac_number batch filenames
CHECK_STR="Cal_fullsim_batch_"
rgi_fns=$(find $CHECK_STR*)
echo rgi_glac_number filenames:$rgi_fns
# create list
list_rgi_fns=($rgi_fns)
echo first_batch:${list_rgi_fns[0]}

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE

cd $SLURM_SUBMIT_DIR
# Generate a list of allocated nodes
NODELIST=`srun -l /bin/hostname | awk '{print $2}' | sort -u`
echo $NODELIST

count=0
# Launch the application
for NODE in $NODELIST; do
  #srun --nodes=1 -w $NODE ./test.sh &
  rgi_fn=${list_rgi_fns[count]}
  echo $rgi_fn
  srun --exclusive -N1 -n1 python run_calibration_woggm.py -num_simultaneous_processes=24 -rgi_glac_number_fn=$rgi_fn&
  echo $NODE
  echo $count
  ((count++))
done
wait