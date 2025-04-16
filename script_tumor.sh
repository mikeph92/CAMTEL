#!/bin/bash

# Define common parameters
CLASSIFICATION_TASK="tumor"
CROP_SIZE=96
TESTSETS=("pannuke" "ocelot" "nucls")
DATASET_PATH="dataset/tumor_dataset.csv"
CLUSTER_PATH="clustering/output"
INFERENCE_PATH="clustering_updated/inference_results"

# Loop over each test set
for TESTSET in "${TESTSETS[@]}"; do
    echo "Processing $TESTSET..."

    # # Run training with multitask enabled
    # python train_no_aug.py \
    #     --dataset_path "$DATASET_PATH" \
    #     --cluster_path "$CLUSTER_PATH" \
    #     --classification_task "$CLASSIFICATION_TASK" \
    #     --crop_size "$CROP_SIZE" \
    #     --testset "$TESTSET" \
    #     --multitask \
    #     --use_amp

    # Run testing with multitask enabled and multiple inference methods
    python test_no_aug.py \
        --dataset_path "$DATASET_PATH" \
        --inference_path "$INFERENCE_PATH" \
        --classification_task "$CLASSIFICATION_TASK" \
        --crop_size "$CROP_SIZE" \
        --testset "$TESTSET" \
        --multitask \
        --use_amp \
        --inference_methods cluster weighted_voting weighted_sum

    echo "Finished $TESTSET."
    echo "#######################################"
done

echo "All experiments completed."
