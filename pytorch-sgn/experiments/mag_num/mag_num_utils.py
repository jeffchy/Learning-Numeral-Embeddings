from word2number.w2n import word_to_num

def detect_number(idx2word):
    """
    :param idx2word: idx2word list
    :return new_dict:
    """

    out_dict = {}

    for i in range(len(idx2word)):
        try:
            num = word_to_num(idx2word[i])
            if num not in out_dict:
                out_dict[float(num)] = i

        except ValueError:
            pass

    sorted_key =  sorted(out_dict)
    new_dict = {}
    for k in sorted_key:
        new_dict[k] = out_dict[k]


    return new_dict
