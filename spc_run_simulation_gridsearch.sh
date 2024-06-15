#!/bin/bash
#
# filename: 
#
# Example SLURM script to run a job.
# The lines beginning #SBATCH set various queuing parameters.
#

#SBATCH -J -spc_run_gs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=126
#SBATCH -t 46:00:00
#SBATCH --mem-per-cpu=16000MB
#SBATCH --output=/trace/group/rounce/btober/misc/jobs/oib-ak/out/%j.out

# activate desired environment
module load aocc/3.2.0
module load anaconda3/2021.05
source activate /trace/home/btober/.conda/envs/pygem-dev

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS
echo cpus_per_task: $SLURM_CPUS_PER_TASK
echo rgi_region01: $1

# define variables
startyr=1990
endyr=2022
export num_proc=$SLURM_CPUS_PER_TASK # Size of multiprocessing pool
start=$SECONDS

# from Rounce et al., 2023:
# Kennicott
# kp range: [0.293,3.022]
# tbias range: [-3.017,4.249]
# ddfsnow range: [0.001,0.007]

# Root
# kp range: [0.124,3.655]
# tbias range: [-2.646,4.866]
# ddfsnow range: [0.002,0.009]

# define parameter ranges based on above
kp=$(seq 0.1 .1 5)
# temperature bias (tbias) - [C]
tbias=$(seq -4 .1 6)
# degree day factor of snot (ddfsnow) - [m w.e. d^-1 C^-1]
ddfsnow=$(seq 0.001 .0005 .009)

# list of glacier numbers to run
glac_nums=(1.22193 1.15645 1.15769)
nglacs=${#glac_nums[@]}
# int division to get num copies
ncopies=$((num_proc / nglacs))
# update num_procs
num_proc=$((ncopies * nglacs))

# make $num_proc duplicates of each glacier directory
for glac_n in ${glac_nums[@]}; do
    python /trace/group/rounce/btober/PyGEM-scripts/duplicate_gdirs.py -rgi_glac_number $glac_n -num_copies $ncopies
done

# root path to oggm glacier direcories
oggm_fp=/trace/group/rounce/shared/OGGM/gdirs_

touch params.txt
rm params.txt
touch params.txt

i=1
# build parameter set text file that will be passed to python call as CLI
for kp_ in ${kp[@]}; do
    for tbias_ in ${tbias[@]}; do
        for ddfsnow_ in ${ddfsnow[@]}; do
                # get core/job number based on iteration
                core_number=$(( (i - 1) % ncopies + 1 ))
                for glac_n in ${glac_nums[@]}; do
                    echo python /trace/group/rounce/btober/PyGEM-scripts/run_simulation.py -gcm_name ERA5 -rgi_glac_number $glac_n -gcm_startyear $startyr -gcm_endyear $endyr -tbias $tbias_ -kp $kp_ -ddfsnow $ddfsnow_ -oggm_working_dir $oggm_fp$core_number/ >> params.txt
                done
                ((i += 1))  # Increment the incrementer by 1
        done
    done
done

# Feed parameter sets to GNU
parallel -J $num_proc :::: params.txt

echo "TOTAL RUNTIME: "$(($SECONDS / 3600)) hrs $((($SECONDS / 60) % 60)) min $(($SECONDS % 60)) sec $((i)) iterations""

mv /trace/group/rounce/btober/misc/jobs/oib-ak/out/$SLURM_JOB_ID.out /trace/group/rounce/btober/misc/jobs/oib-ak/out/grid_search.out