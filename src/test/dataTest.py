'''
 * @author Waileinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-07 15:52:24
 * @modify date 2020-08-07 15:52:24
'''

import sys
import os
sys.path.append(os.path.join(os.getcwd(),'src'))
from data import build_corpus
# import data
import torch



def Test_build_corpus():
    word_lists,tag_lists,word2id,tag2id=build_corpus('train')
    print("Training item number:{},tag number :{}, word number:{}"\
        .format(len(word_lists),len(tag2id),len(word2id)))
    print("Data Example:")
    for i in range(10):
        word_list=word_lists[i]
        tag_list=tag_lists[i]
        sent=''.join(word_list)
        tag_sent=''.join(tag_list)
        print(sent)
        print(tag_sent)
    print(tag2id)


if __name__=='__main__':
    # print(sys.path)
    # print(__name__)
    Test_build_corpus()