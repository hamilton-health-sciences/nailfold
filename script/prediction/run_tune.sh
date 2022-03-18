#!/bin/bash

GPUSPEC=gpu:tesla:4
# GPUSPEC=gpu:6

if [ -z "$3" ]; then
    OUTCOME=$1
    REPETITION=$2
else
    OUTCOME=$1
    CONDITIONAL=$2
    REPETITION=$3
fi

echo "Current path (should be set to repository root): `pwd`"

if [ -z "$CONDITIONAL" ]; then
    OUTCOME_STRING=$OUTCOME
else
    OUTCOME_STRING="$OUTCOME (conditional on $CONDITIONAL)"
fi

if [ ! -z "$CONDITIONAL" ]; then
    ADDITIONAL_ARGS="--condition_on $CONDITIONAL"
fi

echo Running tuning script for outcome $OUTCOME_STRING repetition $REPETITION, \
     you have 5 seconds to cancel...

sleep 5

for fold in `seq 0 4`; do
    PYTHONPATH=`pwd` CUBLAS_WORKSPACE_CONFIG=:4096:8 SLURM_JOB_NAME=bash \
        srun -c 24 --mem=200G --gres=${GPUSPEC} --pty \
        python3 script/prediction/tune.py \
        --outcome $OUTCOME --repetition $REPETITION --fold $fold \
        $ADDITIONAL_ARGS
done
