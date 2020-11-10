from glob import glob
import re

if __name__ == '__main__':
    path = '../../data/wikipedia/enwiki-latest-small-p10p30302-with-numerals'
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()

    RE_NUMBER = re.compile(r"(\d+(,\d{3})*(\.\d+)?)", re.UNICODE)
    RE_TOKEN = re.compile(r"((\d+(,\d{3})*(\.\d+)?)|([^\W\d]+)|(\S))", re.UNICODE)
    re_result = re.findall(RE_NUMBER, s)
    print(len(re_result))
    print(re_result[:1000])

    print(len(s.split()))