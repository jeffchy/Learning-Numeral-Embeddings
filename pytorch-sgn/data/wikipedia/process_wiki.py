import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora import WikiCorpus
import argparse
import re
import glob
import os

# RE_TOKEN = re.compile(r"((\d+(,\d{3})*(\.\d+)?)|([^\W\d]+)|(\S))", re.UNICODE) # normal number
# RE_TOKEN = re.compile(r"(((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?)|([^\W\d]+)|(\S))") # number, negative, fraction
RE_TOKEN = re.compile(r"(((\w+-\d+)|(\w+-\w+)|((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))|([^\W\d]+)|(\S))", re.UNICODE)

all_token_num = 0

def custom_tokenizer(text, token_min_len=1, token_max_len=20, lower=True):
	tokens = []
	for token in RE_TOKEN.findall(text):
		if isinstance(token, tuple):
			token = token[0]

		if len(token) > token_max_len or len(token) < token_min_len:
			pass

		else:
			if lower:
				token = token.lower()

			tokens.append(token)

	return tokens


def make_corpus(in_f, out_f):

	"""Convert Wikipedia xml dump file to text corpus"""
	global all_token_num	
	output = open(out_f, 'w', encoding='utf-8')
	wiki = WikiCorpus(in_f, tokenizer_func=custom_tokenizer)

	i = 0
	for text in wiki.get_texts():
		all_token_num += len(text)
		output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
		i = i + 1
		if (i % 10000 == 0):
			print('Processed ' + str(i) + ' articles, ' + str(all_token_num) + 'tokens in total')
			
	output.close()
	print('Processing complete!')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, help="wiki dumps file path")
	parser.add_argument('--output', type=str, help="processed file path")
	args = parser.parse_args()
	infile = args.input
	outfile = args.output
	if infile[-3:] == 'bz2':
		make_corpus(infile, outfile)

	elif os.path.isdir(infile):
		files = glob.glob(infile+'/*.bz2')	
		print(files)
		for f in files:
			print('processing file: {}'.format(f), 'num of tokens now: {}'.format(all_token_num))
			make_corpus(f, f[:-3]+'txt')
		

