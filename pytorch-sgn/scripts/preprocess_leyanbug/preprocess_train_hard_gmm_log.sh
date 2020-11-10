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
echo "train som $var rd"
python preprocess_leyan.py --save_dir=data/leyanbug --mode=train_gmm --num_components=$var --gmm_init_mode=rd --gmm_type=hard --log_space

echo "train som $var km"
python preprocess_leyan.py --save_dir=data/leyanbug --mode=train_gmm --num_components=$var --gmm_init_mode=km --gmm_type=hard --log_space

echo "train som $var fp"
python preprocess_leyan.py --save_dir=data/leyanbug --mode=train_gmm --num_components=$var --gmm_init_mode=fp --prototype_path=prototypes-$var-0.5-1.0.dat --gmm_type=hard --log_space

done
