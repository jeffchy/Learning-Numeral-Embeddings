#!/bin/bash
sizes=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cuda) cuda="$2"; shift 2;;
    --dir) dir="$2"; shift 2;;
    --method) method="$2"; shift 2;;
    --gmm_type) gmm_type="$2"; shift 2;;
    --log_dir) log_dir="$2"; shift 2;;
    *) sizes+=("$1"); shift 1;;
  esac
done

mkdir $log_dir
cd ..

train_portion=0.1

for var in "${sizes[@]}"
do
if [ $method = 'GMM' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-rd-$gmm_type-0 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-rd-$gmm_type-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-rd-$gmm_type-1 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-rd-$gmm_type-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-rd-$gmm_type-2 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-rd-$gmm_type-2 &
fi

if [ $method = 'prototype' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=prototype --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/som-$var-0 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-som-$var-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=prototype --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/som-$var-1 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-som-$var-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=prototype --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/som-$var-2 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-som-$var-2 &
fi

if [ $method = 'token' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=token --preprocess_dir=../../data/$dir/NumeralAsTokenUnkNumeral-550 --word_emb_dir=../../data/$dir/save/NumeralAsTokenUnkNumeral-0-550 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-token-0-550 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=token --preprocess_dir=../../data/$dir/NumeralAsTokenUnkNumeral-550 --word_emb_dir=../../data/$dir/save/NumeralAsTokenUnkNumeral-1-550 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-token-1-550 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=token --preprocess_dir=../../data/$dir/NumeralAsTokenUnkNumeral-550 --word_emb_dir=../../data/$dir/save/NumeralAsTokenUnkNumeral-2-550 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-token-2-550 &
fi



if [ $method = 'GMMFP' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-fp-$gmm_type-0 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-fp-$gmm_type-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-fp-$gmm_type-1 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-fp-$gmm_type-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-fp-$gmm_type-2 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-fp-$gmm_type-2 &
fi

if [ $method = 'GMMKM' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-km-$gmm_type-0 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-km-$gmm_type-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-km-$gmm_type-1 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-km-$gmm_type-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=GMM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/gmm-$var-km-$gmm_type-2 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-gmm-$var-km-$gmm_type-2 &
fi


if [ $method = 'LSTM' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=LSTM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/LSTM-0 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-LSTM-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=LSTM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/LSTM-1 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-LSTM-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=LSTM --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/LSTM-2 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-LSTM-2 &
fi


if [ $method = 'LSTMnew' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=LSTM --preprocess_dir=../../data/$dir/NumeralAsNumeralLSTM --word_emb_dir=../../data/$dir/save/LSTMnew-0 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-LSTMnew-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=LSTM --preprocess_dir=../../data/$dir/NumeralAsNumeralLSTM --word_emb_dir=../../data/$dir/save/LSTMnew-1 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-LSTMnew-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=LSTM --preprocess_dir=../../data/$dir/NumeralAsNumeralLSTM --word_emb_dir=../../data/$dir/save/LSTMnew-2 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-LSTMnew-2 &
fi


if [ $method = 'fixed' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=fixed --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/fixed-0 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-fixed-0 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=fixed --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/fixed-1 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-fixed-1 &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=fixed --preprocess_dir=../../data/$dir/NumeralAsNumeral --word_emb_dir=../../data/$dir/save/fixed-2 --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-fixed-2 &
fi



if [ $method = 'no-embed' ]
then
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=none --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-no-embed-no-train &
CUDA_VISIBLE_DEVICES=$cuda nohup /usr/bin/python3.6 main.py --mode=none --train_embedding=True --train_portion=$train_portion > tag_leyan/$log_dir/tag-out-no-embed-train &
fi

done


