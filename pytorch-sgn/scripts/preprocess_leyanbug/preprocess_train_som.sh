#!/bin/bash
sizes=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    *) sizes+=("$1"); shift 1;;
  esac
done

cd ..
cd ..

for var in "${sizes[@]}"
do
echo "train som $var"
python preprocess_leyan.py --save_dir=data/leyanbug --mode=train_som --num_iters=10000 --num_prototypes=$var --lr=1 --sigma=0.5

done
