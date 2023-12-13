#!/bin/bash

#SBATCH -J BS
#SBATCH -p overflow
#SBATCH -w beauty,nvidia-dgx1
#SBATCH -o file-%A.out
#SBATCH -e file-%A.err
#SBATCH -G 1

#SBATCH -t 18:0:0
#SBATCH --mem 16G
#SBATCH --nodes=1
#SBATCH --mail-type=ALL --mail-user=xmu22@emory.edu

mkdir -p "$HOME/CS534/project/ml_output"
SOURCE_DIR="$HOME/CS534/project"
OUTPUT_FILE="$HOME/CS534/project/ml_output/ml_output.txt"

source "$HOME/env/bin/activate"
echo -e "changing directory to target" > $OUTPUT_FILE
cd "$HOME/CS534/project"
echo -e "Start running" > $OUTPUT_FILE
python ml_model.py > "$OUTPUT_FILE" 2>&1
