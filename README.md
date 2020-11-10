# Learning-Numeral-Embeddings
Source code for the paper "Learning Numeral Embedding" Chengyue Jiang, Zhonglin Nian, Kaihao Guo, Shanbo Chu, Yinggong Zhao, Libin Shen, and Kewei Tu, accepted in Findings of EMNLP, 2020

## Citation
```
@inproceedings{jiang-etal-2020-learning,
    title = "Learning Numeral Embedding",
    author = "Jiang, Chengyue  and
      Nian, Zhonglin  and
      Guo, Kaihao  and
      Chu, Shanbo  and
      Zhao, Yinggong  and
      Shen, Libin  and
      Tu, Kewei",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.235",
    pages = "2586--2599",
    abstract = "Word embedding is an essential building block for deep learning methods for natural language processing. Although word embedding has been extensively studied over the years, the problem of how to effectively embed numerals, a special subset of words, is still underexplored. Existing word embedding methods do not learn numeral embeddings well because there are an infinite number of numerals and their individual appearances in training corpora are highly scarce. In this paper, we propose two novel numeral embedding methods that can handle the out-of-vocabulary (OOV) problem for numerals. We first induce a finite set of prototype numerals using either a self-organizing map or a Gaussian mixture model. We then represent the embedding of a numeral as a weighted average of the prototype number embeddings. Numeral embeddings represented in this manner can be plugged into existing word embedding learning approaches such as skip-gram for training. We evaluated our methods and showed its effectiveness on four intrinsic and extrinsic tasks: word similarity, embedding numeracy, numeral prediction, and sequence labeling.",
}
```


![](/imgs/Prototype.png)
![](/imgs/GMM.png)
![](/imgs/LSTM.png)
![](/imgs/diagram.PNG)

# Run the code (Example)
## Requirements
pytorch==0.4.1 <br>
scikit-learn==0.19.2 <br>
matplotlib==2.2.2 <br>
seaborn==0.9.0 <br>
numpy==1.15.0

## check options
List all options and their explainations, use: 
```
python <...>.py --help
```

## Preprocess
```
python preprocess.py --MAXDATA=20000000 --save_dir=data/wikipedia/preprocess1B --filtered=filtered.txt --corpus=data/wikipedia/wikiraw/bz2/train.txt --max_vocab=300000 --mode=all --window=5 --scheme=numeral_as_numeral --saved_dir_name=NumeralAsNumeral30W 
```
preprocess the original plain text data <--corpus> (train.txt), and write to a filtered plain text file <--filtered> (filtered.txt), then 
generate all necessary files for training in <--save_dir> including vocabularies and training batches.

## Train SOM / GMM
```
python preprocess.py --mode=train_som --num_iters=200000 --num_prototypes=100 --lr=1 --sigma=0.6 --save_dir=data/wikipedia/preprocess1B/ --saved_dir_name=NumeralAsNumeral30W
```
```
python preprocess.py --mode=train_gmm --num_components=200 --gmm_iters=100 --save_dir=data/wikipedia/preprocess1B/ --prototype_path=prototypes-200-0.6-1.0.dat --saved_dir_name=NumeralAsNumeral30W --gmm_type=hard
```

## Train word vectors
```
python train.py --cuda --weights --preprocess_dir=./data/wikipedia/preprocess1B/NumeralAsNumeral30W --save_dir=./data/wikipedia/save/1B30W/prototypes/2-0005 --log_dir=./data/wikipedia/logs/1B30W/prototypes/2-0005 --epoch=1 --n_neg=5 --mb=2048 --scheme=prototype --e_dim=300 --prototypes_path=prototypes-200-0.6-1.0.dat --lr=0.0005
```

## Generate Numeral Embeddings
see the file create_embedding.py in /experiments/probing/src
