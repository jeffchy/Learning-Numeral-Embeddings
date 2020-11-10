cd ..
cd ..
echo "build and convert, NumeralAsNumeral"
/usr/bin/python3.6 preprocess_leyan.py --save_dir=data/leyanbug --filtered=leyanbug.filtered --corpus=data/leyanbug/train_corpus.txt --window=2 --scheme=numeral_as_numeral --mode=build --max_vocab=500
/usr/bin/python3.6 preprocess_leyan.py --save_dir=data/leyanbug --filtered=leyanbug.filtered --corpus=data/leyanbug/train_corpus.txt --window=2 --scheme=numeral_as_numeral --mode=convert --max_vocab=500
echo "build and convert, NumeralAsTokenUnkNumeral"
/usr/bin/python3.6 preprocess_leyan.py --save_dir=data/leyanbug --filtered=leyanbug.filtered --corpus=data/leyanbug/train_corpus.txt --window=2 --scheme=numeral_as_token_unk_numeral --mode=build --max_vocab=550
/usr/bin/python3.6 preprocess_leyan.py --save_dir=data/leyanbug --filtered=leyanbug.filtered --corpus=data/leyanbug/train_corpus.txt --window=2 --scheme=numeral_as_token_unk_numeral --mode=convert --max_vocab=550
echo "build and convert, NumeralAsNumeralLSTM"
/usr/bin/python3.6 preprocess_leyan.py --save_dir=data/leyanbug --filtered=leyanbug.filtered --corpus=data/leyanbug/train_corpus.txt --window=2 --scheme=numeral_as_numeral_lstm --mode=build --max_vocab=500
/usr/bin/python3.6 preprocess_leyan.py --save_dir=data/leyanbug --filtered=leyanbug.filtered --corpus=data/leyanbug/train_corpus.txt --window=2 --scheme=numeral_as_numeral_lstm --mode=convert --max_vocab=500



