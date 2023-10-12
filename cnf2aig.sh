#!/bin/bash
# 遍历data文件夹中的所有文件，将cnf格式转换为aig格式

convert_command="./aiger/cnf2aig/cnf2aig"

train_dir="./data/uf50-218-tvt/train/"
train_output_dir="./aigdata/train/"

validation_dir="./data/uf50-218-tvt/validation/"
validation_output_dir="./aigdata/validation/"

eval_dir="./data/uf50-218-tvt/eval-problems-paths/"
eval_output_dir="./aigdata/eval-problems-paths/"

if [ ! -e $train_output_dir ]
then
mkdir -p $train_output_dir
fi

if [ ! -e $validation_output_dir ]
then
mkdir -p $validation_output_dir
fi

if [ ! -e $eval_output_dir ]
then
mkdir -p $eval_output_dir
fi

for train_file in $(ls $train_dir); do
    train_path=$train_dir$train_file
    output_file=${train_file%.cnf*}.aiger
    train_process_path=$train_output_dir$train_file
    train_output_path=$train_output_dir$output_file
    sed '/^%$/d;/^0$/d' $train_path > $train_process_path
    $convert_command $train_process_path $train_output_path
    rm $train_process_path
done

for validation_file in $(ls $validation_dir); do
    validation_path=$validation_dir$validation_file
    output_file=${validation_file%.cnf*}.aiger
    validation_process_path=$validation_output_dir$validation_file
    validation_output_path=$validation_output_dir$output_file
    sed '/^%$/d;/^0$/d' $validation_path > $validation_process_path
    $convert_command $validation_process_path $validation_output_path
    rm $validation_process_path
done

for eval_file in $(ls $eval_dir); do
    eval_path=$eval_dir$eval_file
    output_file=${eval_file%.cnf*}.aiger
    eval_process_path=$eval_output_dir$eval_file
    eval_output_path=$eval_output_dir$output_file
    sed '/^%$/d;/^0$/d' $eval_path > $eval_process_path
    $convert_command $eval_process_path $eval_output_path
    rm $eval_process_path
done