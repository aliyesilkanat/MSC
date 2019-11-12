#!/bin/bash
#
# CompecTA (c) 2018
#
#
# TODO:
#   - Set name of the job below changing "Keras" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch keras_pulsar_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=AdaptiveVC128segment-256Mel
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=longer
#SBATCH --output=%j-deep.out
#SBATCH --mail-user=ALL
#SBATCH --mail-user=ali.yesilkanat@boun.edu.tr
#SBATCH --mem-per-cpu=40000

################################################################################
#source /etc/profile.d/z_compecta.sh
echo "source /etc/profile.d/z_compecta.sh"
################################################################################

# MODULES LOAD...

#module load cudnn/7.1.1/cuda-9.0
#module load anaconda/3.6

################################################################################

echo ""
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo ""
echo ""

echo "Running Keras-Tensorflow command..."
echo "===================================================================================="
RET=$?

stage=0
segment_size=128
n_out_speakers=20
test_prop=0.1
sample_rate=24000
training_samples=10000000
testing_samples=10000
n_utt_attr=5000


/raid/users/ayesilkanat/anaconda3/bin/python  PreprocessingSetup1.py





RET=$?

echo ""
echo "===================================================================================="
echo "Solver exited with return code: $RET"
exit $RET
