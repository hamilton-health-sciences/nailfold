#!/bin/bash

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
    FULL_OUTCOME_NAME="${OUTCOME}_cond_${CONDITIONAL}"
else
    FULL_OUTCOME_NAME=$OUTCOME
fi

echo Extracting predictions...
PYTHONPATH=. python3 script/prediction/predict.py --outcome $OUTCOME \
    $ADDITIONAL_ARGS --split_seeds $REPETITION

echo Computing prediction evaluation metrics...
PYTHONPATH=. python3 script/prediction/evaluate.py \
    --outcome $FULL_OUTCOME_NAME

if [ "$OUTCOME" != "diabetes" ]; then
    echo Computing evaluation metrics for diabetes subanalysis...
    PYTHONPATH=. python3 script/prediction/evaluate.py \
        --outcome $FULL_OUTCOME_NAME --subanalysis "Diabetes Diagnosis"
fi

if [ -z "$CONDITIONAL" ]; then
    echo Computing saliency maps...
    for fold_idx in `seq 0 4`; do
        PYTHONPATH=. python3 script/analysis/explain.py \
            --outcome $FULL_OUTCOME_NAME --subset test \
            --outcome_value 1. --fold $fold_idx
        PYTHONPATH=. python3 script/analysis/explain.py \
            --outcome $FULL_OUTCOME_NAME --subset test \
            --outcome_value 0. --fold $fold_idx
    done
fi

if [ "$OUTCOME" = "diabetes" ] || [ "$OUTCOME" = "hba1c_high" ] || [ "$OUTCOME" = "hba1c_moderatehigh" ]; then
    echo Computing representations...
    for fold_idx in `seq 0 4`; do
        PYTHONPATH=. python3 script/analysis/infer_representations.py \
            --outcome $OUTCOME $ADDITIONAL_ARGS --repetition $REPETITION \
            --fold $fold_idx
    done

    echo Computing representation/measurement coherence...
    MC_PFX=results/${FULL_OUTCOME_NAME}/measurement_coherence
    echo > ${MC_PFX}_capillary_count
    echo > ${MC_PFX}_capillary_length
    for fold_idx in `seq 0 4`; do
        PYTHONPATH=. python3 script/analysis/measurement_coherence.py \
            --outcome $FULL_OUTCOME_NAME --repetition $REPETITION --fold $fold_idx \
            --measure capillary_count \
            --threshold 10 >> ${MC_PFX}_capillary_count
        PYTHONPATH=. python3 script/analysis/measurement_coherence.py \
            --outcome $FULL_OUTCOME_NAME --repetition $REPETITION --fold $fold_idx \
            --measure lengths --threshold 200 >> ${MC_PFX}_capillary_length
    done
fi
