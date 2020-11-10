import re

RE_TOKEN = re.compile(r"(((\w+-\d+)|(\w+-\w+)|((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))|([^\W\d]+)|(\S))", re.UNICODE)

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


def preprocess_dev(path):
	with open(path,'r',encoding='utf-8') as file:
		s = file.read()
	tokens = custom_tokenizer(s)
	return tokens

if __name__ == '__main__':
	tokens = preprocess_dev('./experimental/age_larger/age_larger.txt')
	s = ' '.join(tokens)
	with open('./experimental/age_larger/age_larger.proc', 'w', encoding='utf-8') as f:
		f.write(s)
