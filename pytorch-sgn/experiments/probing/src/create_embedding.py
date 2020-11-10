import pickle
from experiments.mag_num.MagNumEvaluator import MagNumEvaluator
import os

if __name__ == '__main__':
    # print()
    numerals = pickle.load(open('../mag_num/data/numerals_mag_infre.pkl', 'rb'))
    evaluator = MagNumEvaluator(numerals_dict=numerals, type='MAG')

    print('=== Loading Prototype ===')
    prototype_size = ['2', '3', '5']
    for sz in prototype_size:
        trained_prototypes = pickle.load(
            open('../../data/wikipedia/save/1B30W/prototypes/{}-0005/trained_prototypes_epoch1_{}00_1.0.dat'.format(sz, sz),'rb'))
        evaluator.load_prototype(trained_prototypes)
        pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/p-{}00'.format(sz), 'wb'))

    print('=== Loading Prototype LOG ===')
    prototype_size = ['2', '3', '5']
    for sz in prototype_size:
        trained_prototypes = pickle.load(
            open('../../data/wikipedia/save/1B30W/prototypes_log/{}-0005/trained_prototypes_epoch1_{}00_1.0.dat'.format(
                sz, sz), 'rb'))
        evaluator.load_prototype(trained_prototypes, log_space=True)
        pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/p-log-{}00'.format(sz), 'wb'))

    print('=== Loading GMM-RD-SOFT ===')
    gmms = ['300', '500']
    for gmm_name in gmms:
        trained_prototypes = pickle.load(open('../../data/wikipedia/save/1B30W/gmms/gmm-{}-rd-soft/trained_gmms_epoch1_{}.dat'.format(gmm_name, gmm_name),'rb'))
        gmm = pickle.load(open('../../data/wikipedia/preprocess1B/NumeralAsNumeral30W/gmm/gmm-{}-rd-soft.dat'.format(gmm_name),'rb'))
        evaluator.load_GMM(trained_prototypes, gmm)
        pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/gmm-rd-soft-{}'.format(gmm_name), 'wb'))

    print('=== Loading GMM-RD-SOFT-LOG ===')
    gmms = ['300', '500']
    for gmm_name in gmms:
        trained_prototypes = pickle.load(open('../../data/wikipedia/save/1B30W/gmms_log/gmm-{}-rd-soft/trained_gmms_epoch1_{}.dat'.format(gmm_name, gmm_name),'rb'))
        gmm = pickle.load(open('../../data/wikipedia/preprocess1B/NumeralAsNumeral30W/gmm_log/gmm-{}-rd-soft.dat'.format(gmm_name),'rb'))
        evaluator.load_GMM(trained_prototypes, gmm, log_space=True)
        pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/gmm-log-rd-soft-{}'.format(gmm_name), 'wb'))

    print('=== Loading GMM-RD-HARD ===')
    gmms = ['300', '500']
    for gmm_name in gmms:
        trained_prototypes = pickle.load(open('../../data/wikipedia/save/1B30W/gmms/gmm-{}-rd-hard/trained_gmms_epoch1_{}.dat'.format(gmm_name, gmm_name),'rb'))
        gmm = pickle.load(open('../../data/wikipedia/preprocess1B/NumeralAsNumeral30W/gmm/gmm-{}-rd-hard.dat'.format(gmm_name),'rb'))
        evaluator.load_GMM(trained_prototypes, gmm,)
        pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/gmm-rd-hard-{}'.format(gmm_name), 'wb'))

    print('=== Loading GMM-RD-HARD-LOG ===')
    gmms = ['300', '500']
    for gmm_name in gmms:
        trained_prototypes = pickle.load(open('../../data/wikipedia/save/1B30W/gmms_log/gmm-{}-rd-hard/trained_gmms_epoch1_{}.dat'.format(gmm_name, gmm_name),'rb'))
        gmm = pickle.load(open('../../data/wikipedia/preprocess1B/NumeralAsNumeral30W/gmm_log/gmm-{}-rd-hard.dat'.format(gmm_name),'rb'))
        evaluator.load_GMM(trained_prototypes, gmm, log_space=True)
        pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/gmm-log-rd-hard-{}'.format(gmm_name), 'wb'))

    print('=== Loading LSTMnew Baseline ===')
    idx2word = pickle.load(open('../../data/wikipedia/preprocess1B/NumeralAsNumeralLSTM30W/idx2word.dat','rb'))
    word2idx = pickle.load(open('../../data/wikipedia/preprocess1B/NumeralAsNumeralLSTM30W/word2idx.dat','rb'))
    evaluator.reload(numerals_dict=numerals, idx2word=idx2word, word2idx=word2idx)


    print('=== Loading Fixed Baseline ===')
    idx2vec_i = pickle.load(open('../../data/wikipedia/save/1B30W/FIXED/idx2vec_i_epoch1.dat','rb'))
    evaluator.load_fixed(idx2vec_i=idx2vec_i)
    pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/Fixed', 'wb'))

    print('=== Loading Token Baseline ===')
    idx2word = pickle.load(open('../../data/wikipedia/preprocess1B/NumeralAsTokenUnkNumeral30W/idx2word.dat', 'rb'))
    word2idx = pickle.load(open('../../data/wikipedia/preprocess1B/NumeralAsTokenUnkNumeral30W/word2idx.dat', 'rb'))
    evaluator.reload(numerals_dict=numerals, idx2word=idx2word, word2idx=word2idx)

    idx2vec_o = pickle.load(open('../../data/wikipedia/save/1B30W/token1/idx2vec_o_epoch1.dat', 'rb'))
    idx2vec_i = pickle.load(open('../../data/wikipedia/save/1B30W/token1/idx2vec_i_epoch1.dat', 'rb'))
    evaluator.load_TOKEN(idx2vec_i=idx2vec_i, idx2vec_o=idx2vec_o)
    pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/Token', 'wb'))

    idx2vec_o = pickle.load(open('../../data/wikipedia/save/1B30W/LSTMnew/idx2vec_o_epoch1.dat','rb'))
    idx2vec_i = pickle.load(open('../../data/wikipedia/save/1B30W/LSTMnew/idx2vec_i_epoch1.dat','rb'))
    LSTM_model_path = '../../data/wikipedia/save/1B30W/LSTMnew/sgns_epoch1.pt'
    evaluator.load_LSTM(idx2vec_i, idx2vec_o, LSTM_model_path)
    pickle.dump(evaluator.numeral_embed_i, open('embed_infre_low/LSTMnew', 'wb'))