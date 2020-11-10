import sys
sys.path.append('../../../utils/')
from number_handler import is_numeral, to_numeral
import numpy as np
import random
random.seed(0)

def get_all_target_numerals(data):
    res = [[] for i in range(8)]

    for i in data:
        numeral_str = i[0]
        magnitude = i[2]
        res[magnitude-1].append(numeral_str)

    return res

def random_sample_numerals(res):
    sampled_res = []
    for mags in res:
        if len(mags) > 100:
            sampled_res.append(random.sample(mags, 100))
        else:
            sampled_res.append(mags)

    for i in range(len(sampled_res)):
        sampled_res[i] = [j for j in sampled_res[i] if (float(j) > 0 and float(j) != 0)]

    sampled_res = [np.array([float(j) for j in i]) for i in sampled_res]
    return sampled_res

def create_magnitude_embeds(embeds):
    numeral_embed_i = []
    numeral_embed_o = []
    for i in embeds:

        embed_i = np.array(i[0]).mean(0)
        embed_o = np.array(i[1]).mean(0)

        numeral_embed_i.append(embed_i)
        numeral_embed_o.append(embed_o)

    return np.array(numeral_embed_i), np.array(numeral_embed_o)

