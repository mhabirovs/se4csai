#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

cd /home/u558964/Projects/SE/Final/nmt/

python MT-Preparation/filtering/filter.py WikiMatrix.en-nl.en WikiMatrix.en-nl.nl en nl

python MT-Preparation/subwording/1-train_unigram.py WikiMatrix.en-nl.en-filtered.en WikiMatrix.en-nl.nl-filtered.nl

python MT-Preparation/subwording/2-subword.py source.model target.model WikiMatrix.en-nl.en-filtered.en WikiMatrix.en-nl.nl-filtered.nl

python MT-Preparation/train_dev_split/train_dev_test_split.py 50000 50000 WikiMatrix.en-nl.en-filtered.en.subword WikiMatrix.en-nl.nl-filtered.nl.subword

python Config.py

onmt_build_vocab -config config.yaml -n_sample -1 -num_threads 64

onmt_train -config config.yaml
