'''
 * @author Waileinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-07 15:52:06
 * @modify date 2020-08-07 15:52:06
'''



import random
import pandas as pd
import time
import os
from utils import extend_map,add_label_for_lstmcrf,save_model,flatten_lists,load_model
from models.hmm import HMM
from models.standard import HMM_standard
from models.bilstm_crf import BiLSTM_CRF_Model
from data import build_corpus
from datetime import datetime
from evaluate import Eval_unit,evaluate_entity_label,evaluate_single_label,evaluate_multiclass,unitstopd
from config import ModelPathConfig,ResultPathConfig

import torch
torch.cuda.set_device(1)
cwd=os.getcwd()

def sample_print_test(word_list,tag_list,sample_num=5):
    indices=random.sample(range(len(word_list)),sample_num)
    print_word_list=[word_list[i] for i in indices]
    print_tag_list=[tag_list[i] for i in indices]

    for i in range(sample_num):
        s=' '.join(print_word_list[i])
        s+='\n'
        s+=' '.join(print_tag_list[i])
        print(s)


def bilstm_crf_test(if_train=False):
    model_is_existed=os.path.exists(ModelPathConfig.bilstm_crf)

    print("upload data!")
    word_lists,tag_lists,word2id,tag2id=build_corpus("train")
    test_word_lists,test_tag_lists,_,_=build_corpus("test")

    dev_indices=random.sample(range(len(word_lists)),len(word_lists)//5)
    train_indices=[i for i in range(len(word_lists)) if i not in dev_indices]

    dev_word_lists=[ word_lists[ind] for ind in dev_indices]
    dev_tag_lists=[tag_lists[ind] for ind in dev_indices]
    train_word_lists=[word_lists[ind] for ind in train_indices]
    train_tag_lists=[tag_lists[ind] for ind in train_indices]
    test_word_lists,test_tag_lists=add_label_for_lstmcrf(test_word_lists,test_tag_lists,test=True)
    bilstm_crf_word2id,bilstm_crf_tag2id=extend_map(word2id,tag2id,crf=True)
    if if_train or not model_is_existed:
        print('start to training')
        bilstm_crf_word2id,bilstm_crf_tag2id=extend_map(word2id,tag2id,crf=True)

        train_word_lists,train_tag_lists=add_label_for_lstmcrf(train_word_lists,train_tag_lists,test=False)
        dev_word_lists,dev_tag_lists=add_label_for_lstmcrf(dev_word_lists,dev_tag_lists,test=False)

        # sample_print_test(train_word_lists,train_tag_lists)

        start=datetime.now()
        vocab_size=len(word2id)
        out_size=len(tag2id)

        bilstm_model=BiLSTM_CRF_Model(vocab_size,out_size,crf=True)
        bilstm_model.train(train_word_lists,train_tag_lists,\
            bilstm_crf_word2id,bilstm_crf_tag2id,dev_word_lists,dev_tag_lists)
        deltatime=datetime.now()-start

        print("Training is finished, {} second".format(deltatime.seconds))
        save_model(bilstm_model,ModelPathConfig.bilstm_crf)
        print("Save the model")
    else:
        print("load model")
        bilstm_model=load_model(ModelPathConfig.bilstm_crf)


    print("test the model")
    pred_tag_lists,label_tag_lists,=bilstm_model.test(test_word_lists,test_tag_lists,bilstm_crf_word2id,bilstm_crf_tag2id)


    units=evaluate_entity_label(pred_tag_lists,label_tag_lists,list(tag2id.keys()))
    df=unitstopd(units)
    df.to_csv(ResultPathConfig.bilstm_crf_entity)

    units=evaluate_single_label(pred_tag_lists,label_tag_lists,list(tag2id.keys()))
    df=unitstopd(units)
    df.to_csv(ResultPathConfig.bilstm_crf_model)


def HMM_test(if_train=True):
    model_is_existed=os.path.exists(ModelPathConfig.hmm)

    print("upload data!")
    word_lists,tag_lists,word2id,tag2id=build_corpus("train")
    test_word_lists,test_tag_lists,_,_=build_corpus("test")
    # word_lists,tag_lists,word2id,tag2id=build_corpus("train",data_dir=os.path.join(os.getcwd(),"data",'ResumeNER'))
    # test_word_lists,test_tag_lists,_,_=build_corpus("test",data_dir=os.path.join(os.getcwd(),"data",'ResumeNER'))

    hmm_model=HMM(len(tag2id),len(word2id))

    if if_train or not model_is_existed:
        print("start to training")
        hmm_model.train(word_lists,tag_lists,word2id,tag2id)
        print("save the model")
        save_model(hmm_model,ModelPathConfig.hmm)
    else:
        print("load model")
        hmm_model=load_model(ModelPathConfig.hmm)
    pred_tag_lists=hmm_model.test(test_word_lists,_,word2id,tag2id)
    label_tag_lists=test_tag_lists

    units=evaluate_entity_label(pred_tag_lists,label_tag_lists,list(tag2id.keys()))
    df=unitstopd(units)
    df.to_csv(ResultPathConfig.hmm_entity)

    units=evaluate_single_label(pred_tag_lists,label_tag_lists,list(tag2id.keys()))
    df=unitstopd(units)
    df.to_csv(ResultPathConfig.hmm_model)


def HMM_test_standard(if_train=True):
    model_is_existed=os.path.exists(ModelPathConfig.hmm_standard)

    print("upload data!")
    word_lists,tag_lists,word2id,tag2id=build_corpus("train",data_dir=os.path.join(os.getcwd(),"data",'ResumeNER'))
    test_word_lists,test_tag_lists,_,_=build_corpus("test",data_dir=os.path.join(os.getcwd(),"data",'ResumeNER'))

    hmm_model=HMM_standard(len(tag2id),len(word2id))

    if if_train or not model_is_existed:
        print("start to training")
        hmm_model.train(word_lists,tag_lists,word2id,tag2id)
        print("save the model")
        save_model(hmm_model,ModelPathConfig.hmm_standard)
    else:
        print("load model")
        hmm_model=load_model(ModelPathConfig.hmm_standard)

    pred_tag_lists=hmm_model.test(test_word_lists,word2id,tag2id)
    label_tag_lists=test_tag_lists

    units=evaluate_entity_label(pred_tag_lists,label_tag_lists,list(tag2id.keys()))
    df=unitstopd(units)
    df.to_csv(ResultPathConfig.hmm_entity_standard)

    units=evaluate_single_label(pred_tag_lists,label_tag_lists,list(tag2id.keys()))
    df=unitstopd(units)
    df.to_csv(ResultPathConfig.hmm_model_standard)


if __name__=='__main__':
    bilstm_crf_test(False)



