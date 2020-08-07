
from model import BiLSTM_CRF_Model
from data import build_corpus
import random
from utils import extend_map,add_label_for_lstmcrf,save_model,flatten_lists
import time
from datetime import datetime
import os

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
    

def bilstm_crf_test():
    print("upload data!")
    word_lists,tag_lists,word2id,tag2id=build_corpus("train")
    test_word_lists,test_tag_lists,_,_=build_corpus("test")

    dev_indices=random.sample(range(len(word_lists)),len(word_lists)//5)
    train_indices=[i for i in range(len(word_lists)) if i not in dev_indices]
    
    dev_word_lists=[ word_lists[ind] for ind in dev_indices]
    dev_tag_lists=[tag_lists[ind] for ind in dev_indices]
    train_word_lists=[word_lists[ind] for ind in train_indices]
    train_tag_lists=[tag_lists[ind] for ind in train_indices]

    print('start to training')
    bilstm_crf_word2id,bilstm_crf_tag2id=extend_map(word2id,tag2id,crf=True)

    train_word_lists,train_tag_lists=add_label_for_lstmcrf(train_word_lists,train_tag_lists,test=False)
    dev_word_lists,dev_tag_lists=add_label_for_lstmcrf(dev_word_lists,dev_tag_lists,test=False)
    test_word_lists,test_tag_lists=add_label_for_lstmcrf(test_word_lists,test_tag_lists,test=True)
    
    # sample_print_test(train_word_lists,train_tag_lists)

    start=datetime.now()
    vocab_size=len(word2id)
    out_size=len(tag2id)

    bilstm_model=BiLSTM_CRF_Model(vocab_size,out_size,crf=True)
    bilstm_model.train(train_word_lists,train_tag_lists,\
        bilstm_crf_word2id,bilstm_crf_tag2id,dev_word_lists,dev_tag_lists)
    deltatime=datetime.now()-start

    print("Training is finished, {} second".format(deltatime.seconds))
    model_name="bilstm_crf"
    model_path=os.path.join(cwd,'data','model',model_name)
    
    save_model(bilstm_model,model_path)
    print("Save the model")

    print("test the model")
    pred_tag_lists,label_tag_lists,=bilstm_model.test(test_word_lists,test_tag_lists,bilstm_crf_word2id,bilstm_crf_tag2id)
    
    pred_tag_lists=flatten_lists(pred_tag_lists)
    label_tag_lists=flatten_lists(pred_tag_lists)

    

    

    

    
    
if __name__=='__main__':
    bilstm_crf_test()
    


