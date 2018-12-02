#!/bin/bash
# Yaw misalignment calibrator training and testing script.
# Author: Inwoo Chung (gutomitai@gmail.com)

RAW_DATA_PATH="$1"
OUTPUT_FILE_NAME="$2"

python ymc.py --mode train_test --raw_data_path RAW_DATA_PATH --output_file_name OUTPUT_FILE_NAME \
--num_seq1 72 --num_seq2 1 --gru1_dim 32 --gru2_dim 32 --num_layers 4 --dense1_dim 32 --dropout1_rate 0.8 \
--lr 0.0001 --beta_1 0.9 --beta_2 0.99 --decay 0.00 --epochs 180 --batch_size 2048 --val_ratio 0.001 --model_load 0