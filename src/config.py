'''
 * @author Waileinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-05 16:34:32
 * @modify date 2020-08-05 16:34:32
'''

import os
cwd=os.getcwd()

class LSTMConfig(object):
    #词向量维度
    embedding_size=128

    #LSTM隐藏层维度
    hidden_size=128


class TrainingConfig(object):

    #batch
    batch_size=64
    lr=0.001
    epoches=30
    print_step=5


class ModelPathConfig(object):
    root=os.path.join(cwd,'data','model')
    bilstm_crf=os.path.join(root,'bilstm_crf')
    bilstm=os.path.join(root,'bilstm')
    hmm=os.path.join(root,'hmm')
    hmm_standard=os.path.join(root,'hmm_standard')
    bert=os.path.join(root,'bert')


class ResultPathConfig(object):
    root=os.path.join(cwd,'data','result')
    bilstm_crf_model=os.path.join(root,'bilstm_crf_model_result.csv')
    bilstm_crf_entity=os.path.join(root,'bilstm_crf_entity_result.csv')

    hmm_model=os.path.join(root,'hmm_model_result.csv')
    hmm_entity=os.path.join(root,'hmm_entity_result.csv')


    hmm_model_standard=os.path.join(root,'hmm_model_standard_result.csv')
    hmm_entity_standard=os.path.join(root,'hmm_entity_standard_result.csv')

    bert_model=os.path.join(root,'bert_model_result.csv')
    bert_entity=os.path.join(root,'bert_entity_result.csv')