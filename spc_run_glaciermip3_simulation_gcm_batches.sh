#!/bin/sh
#SBATCH --partition=debug
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24

STARTNO="1"
ENDNO="24"

# region
REGNO="11"
MERGE_SWITCH=0
ORDERED_SWITCH=0

# GCM list
GCM_NAMES_LST="ukesm1-0-ll ipsl-cm6a-lr mri-esm2-0 mpi-esm1-2-hr gfdl-esm4"

# activate environment
module load lang/Anaconda3/5.3.0
source activate oggm_env_v02


# split glaciers into batches for different nodes
python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -option_ordered=$ORDERED_SWITCH -startno=$STARTNO -endno=$ENDNO -regno=$REGNO

# list rgi_glac_number batch filenames
CHECK_STR="R${REGNO}_rgi_glac_number_${STARTNO}-${ENDNO}glac_batch"
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


for GCM_NAME in $GCM_NAMES_LST; do
  GCM_NAME_NOSPACE="$(echo -e "${GCM_NAME}" | tr -d '[:space:]')"
  echo -e "\n$GCM_NAME"

  count=0
  # Launch the application
  for NODE in $NODELIST; do
    #srun --nodes=1 -w $NODE ./test.sh &
    rgi_fn=${list_rgi_fns[count]}
    echo $rgi_fn
    # ONLY WORKS WITH EXCLUSIVE!
    srun --exclusive -N1 -n1 python glaciermip3_run_simulation_woggm.py -gcm_name="$GCM_NAME_NOSPACE" -num_simultaneous_processes=24 -rgi_glac_number_fn=$rgi_fn -option_ordered=$ORDERED_SWITCH &
    ((count++))
  done
  wait
  
done
wait

echo -e "\nScript finished"
