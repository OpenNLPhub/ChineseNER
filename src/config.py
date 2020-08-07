'''
 * @author Waileinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-05 16:34:32
 * @modify date 2020-08-05 16:34:32
'''

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