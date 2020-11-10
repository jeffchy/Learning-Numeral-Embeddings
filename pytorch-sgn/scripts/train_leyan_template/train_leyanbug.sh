#!/bin/bash
sizes=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cuda) cuda="$2"; shift 2;;
    --dir) dir="$2"; shift 2;;
    --method) method="$2"; shift 2;;
    --epoch) epoch="$2"; shift 2;;
    --mb) mb="$2"; shift 2;;
    --lr) lr="$2"; shift 2;;
    --gmm_type) gmm_type="$2"; shift 2;;
    *) sizes+=("$1"); shift 1;;
  esac
done

cd ..
cd ..

for var in "${sizes[@]}"
do
if [ $method = 'GMM' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-rd-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-rd-$gmm_type-0 --log_dir=data/$dir/logs/gmm-$var-rd-$gmm_type-0 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-rd-$gmm_type-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-rd-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-rd-$gmm_type-1 --log_dir=data/$dir/logs/gmm-$var-rd-$gmm_type-1 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-rd-$gmm_type-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-rd-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-rd-$gmm_type-2 --log_dir=data/$dir/logs/gmm-$var-rd-$gmm_type-2 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-rd-$gmm_type-2 &
fi

if [ $method = 'prototype' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=prototype --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/som-$var-0 --log_dir=data/$dir/logs/som-$var-0 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-som-$var-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=prototype --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/som-$var-1 --log_dir=data/$dir/logs/som-$var-1 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-som-$var-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=prototype --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/som-$var-2 --log_dir=data/$dir/logs/som-$var-2 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-som-$var-2 &
fi

if [ $method = 'token' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=none --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/NumeralAsTokenUnkNumeral-0-550 --log_dir=data/$dir/logs/NumeralAsTokenUnkNumeral-0 --preprocess_dir=data/$dir/NumeralAsTokenUnkNumeral-550 > scripts/train_$dir/out-token-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=none --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/NumeralAsTokenUnkNumeral-1-550 --log_dir=data/$dir/logs/NumeralAsTokenUnkNumeral-1 --preprocess_dir=data/$dir/NumeralAsTokenUnkNumeral-550 > scripts/train_$dir/out-token-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=none --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/NumeralAsTokenUnkNumeral-2-550 --log_dir=data/$dir/logs/NumeralAsTokenUnkNumeral-2 --preprocess_dir=data/$dir/NumeralAsTokenUnkNumeral-550 > scripts/train_$dir/out-token-2 &
fi

if [ $method = 'fixed' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=fixed --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/fixed-0 --log_dir=data/$dir/logs/fixed-0 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-fixed-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=fixed --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/fixed-1 --log_dir=data/$dir/logs/fixed-1 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-fixed-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=fixed --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/fixed-2 --log_dir=data/$dir/logs/fixed-2 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-fixed-2 &
fi




if [ $method = 'GMMFP' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-fp-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-fp-$gmm_type-0 --log_dir=data/$dir/logs/gmm-$var-fp-$gmm_type-0 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-fp-$gmm_type-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-fp-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-fp-$gmm_type-1 --log_dir=data/$dir/logs/gmm-$var-fp-$gmm_type-1 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-fp-$gmm_type-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-fp-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-fp-$gmm_type-2 --log_dir=data/$dir/logs/gmm-$var-fp-$gmm_type-2 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-fp-$gmm_type-2 &
fi

if [ $method = 'GMMKM' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-km-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-km-$gmm_type-0 --log_dir=data/$dir/logs/gmm-$var-km-$gmm_type-0 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-km-$gmm_type-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-km-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-km-$gmm_type-1 --log_dir=data/$dir/logs/gmm-$var-km-$gmm_type-1 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-km-$gmm_type-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=GMM --gmms_path=gmm_posterior-$var-km-$gmm_type.dat --save_dir=data/$dir/save/gmm-$var-km-$gmm_type-2 --log_dir=data/$dir/logs/gmm-$var-km-$gmm_type-2 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-gmm-$var-km-$gmm_type-2 &
fi



if [ $method = 'LSTM' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=LSTM --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/LSTM-0 --log_dir=data/$dir/logs/LSTM-0 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-LSTM-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=LSTM --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/LSTM-1 --log_dir=data/$dir/logs/LSTM-1 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-LSTM-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=LSTM --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/LSTM-2 --log_dir=data/$dir/logs/LSTM-2 --preprocess_dir=data/$dir/NumeralAsNumeral > scripts/train_$dir/out-LSTM-2 &
fi

if [ $method = 'LSTMnew' ]
then 
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=LSTM --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/LSTMnew-0 --log_dir=data/$dir/logs/LSTMnew-0 --preprocess_dir=data/$dir/NumeralAsNumeralLSTM > scripts/train_$dir/out-LSTM-new-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=LSTM --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/LSTMnew-1 --log_dir=data/$dir/logs/LSTMnew-1 --preprocess_dir=data/$dir/NumeralAsNumeralLSTM > scripts/train_$dir/out-LSTM-new-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 train.py --cuda --weights --epoch=$epoch --n_neg=5 --mb=$mb --e_dim=50 --alpha=1.0 --no_subsample --lr=$lr --scheme=LSTM --prototypes_path=prototypes-$var-0.5-1.0.dat --save_dir=data/$dir/save/LSTMnew-2 --log_dir=data/$dir/logs/LSTMnew-2 --preprocess_dir=data/$dir/NumeralAsNumeralLSTM > scripts/train_$dir/out-LSTM-new-2 &
fi



done
