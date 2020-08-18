'''
 * @author Waileinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-07 15:52:18
 * @modify date 2020-08-07 15:52:18
'''

import os
from codecs import open
from torch.utils.data import Dataset

def build_corpus(prefix_file,data_dir=os.path.join(os.getcwd(),"data")):
    assert prefix_file in ['train','dev','test']

    word_lists=[]
    tag_lists=[]

    with open(os.path.join(data_dir,prefix_file+"_data"),'r',encoding='utf-8') as f:
        word_list=[]
        tag_list=[]
        for line in f.readlines():
            if line!='\n':
                word,tag=line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                #样本平衡
                # if len(set(tag_list))!=1:
                #     word_lists.append(word_list)
                #     tag_lists.append(tag_list)
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list=[]
                tag_list=[]
    #对Word 和 tag进行编码
    word2id=build_map(word_lists)
    tag2id=build_map(tag_lists)

    return word_lists,tag_lists,word2id,tag2id


def build_map(lists):
    maps={}
    for sent in lists:
        for word in sent:
            if word not in maps:
                maps[word]=len(maps)
    return maps



class NERDataSet(Dataset):

    def __init__(self,prefix_file):
        self.word_lists,self.tag_lists=build_corpus(prefix_file)

    def __len__(self):
        return len(self.word_lists)
    
    def __getitem__(self,idx):
        return {'word_list':self.word_lists[idx],'tag_list':self.tag_lists[idx]}