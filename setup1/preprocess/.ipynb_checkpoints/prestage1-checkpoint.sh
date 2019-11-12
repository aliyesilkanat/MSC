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
#SBATCH --job-name=Prestage1-AdaIn
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --output=%j-deep.out
#SABTCH --mail=ALL
#SBATCH --mail-user=ali.yesilkanat@boun.edu.tr

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
data_dir=/raid/users/ayesilkanat/MSC/new-adaptive-vc/trimmed_vctk_spectrograms/sr_24000_mel_norm
raw_data_dir=/raid/users/ayesilkanat/MSC/VCTK/VCTK-Corpus
n_out_speakers=20
test_prop=0.1
sample_rate=24000
training_samples=10000000
testing_samples=10000
n_utt_attr=5000


/raid/users/ayesilkanat/anaconda3/bin/python make_datasets_vctk.py $raw_data_dir/wav48 $raw_data_dir/speaker-info.txt $data_dir $n_out_speakers $test_prop $sample_rate $n_utt_attr





RET=$?

echo ""
echo "===================================================================================="
echo "Solver exited with return code: $RET"
exit $RET
