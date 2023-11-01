#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

cd /home/u558964/Projects/SE/Final/nmt/

onmt_translate -model models/model.dutch_step_70000.pt -src WikiMatrix.en-nl.en-filtered.en.subword.test -output WikiMatrix.nl.translated -gpu 0 -min_length 1

python3 MT-Preparation/subwording/3-desubword.py target.model WikiMatrix.nl.translated

python3 MT-Preparation/subwording/3-desubword.py target.model WikiMatrix.en-nl.nl-filtered.nl.subword.test
