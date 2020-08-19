'''
 * @author Waileinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-07 15:52:11
 * @modify date 2020-08-07 15:52:11
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

""" Preprocessing Function """

#打入PAD 和 UNK
#打入<start> 和 <end>

""" preprocess the data"""
def extend_map(word2id,tag2id,crf=True):
    word2id['<unk>']=len(word2id)
    word2id['<pad>']=len(word2id)
    tag2id['<unk>']=len(tag2id)
    tag2id['<pad>']=len(tag2id)

    if crf:
        word2id['<start>']=len(word2id)
        word2id['<end>']=len(word2id)
        tag2id['<start>']=len(tag2id)
        tag2id['<end>']=len(tag2id)
    return word2id,tag2id

def add_label_for_lstmcrf(word_lists,tag_lists,test=False):
    assert len(word_lists)==len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append('<end>')
        if not test:
            tag_lists[i].append('<end>')
    return word_lists,tag_lists

#用于将最后[B,L]的结果变成一维，用于计算结果
def flatten_lists(lists):
    ans=[]
    for l in lists:
        if type(l)==list:
            ans+=l
        else:
            ans.append(l)
    return ans;


"""Save and load"""

def save_model(model,file_name):
    """Save the model"""
    with open(file_name,"wb") as f:
        pickle.dump(model,f)


def load_model(file_name):
    """Load the model"""
    with open(file_name,"rb") as f:
        model=pickle.load(f)
    return model


"""   Helper Function for LSTM_CRF Model  """

def tensorized(batch,maps):
    PAD=maps.get('<pad>')
    UNK=maps.get('<unk>')

    #最长序列
    max_len=len(batch[0])
    batch_size=len(batch)

    #初始化为PAD
    batch_tensor=torch.ones(batch_size,max_len).long() * PAD

    #不在词典中的词搭上UNK,其余按照word_to_ix进行标记
    for i,l in enumerate(batch):
        for j,e in enumerate(l):
            batch_tensor[i][j]=maps.get(e,UNK)

    #batch中每个item的长度
    lengths=[len(l) for l in batch]

    return batch_tensor,lengths

def sort_by_lengths(word_lists,tag_lists):
    #按照句子长度进行排序
    assert len(word_lists)==len(tag_lists)
    pairs=list(zip(word_lists,tag_lists))
    indices=sorted(range(len(word_lists)),key=lambda k:len(pairs[k][0]),reverse=True)

    #pairs sorted
    pairs=[ pairs[i] for i in indices]

    word_lists,tag_lists=list(zip(*pairs))

    return word_lists,tag_lists,indices

def cal_loss(result,targets,tag2id):
    '''
    result [B,L,out_size]
    targets:[B,L]
    lengths:[B]
    '''
    PAD= tag2id.get('<pad>')
    assert PAD is not None
    mask=(targets!=PAD) #[B,L] True or False
    targets=targets[mask] #[B*L,1]

    out_size=result.size(2)
    masked_code=mask.unsqueeze(2).expand(-1,-1,out_size) # unsqueeze [B,L,1] expand(-1,-1,out_size) [B,L,out_size]
    result=result.masked_select(mask).contiguous().view(-1,out_size) #masked_select [B*L*out_size,] contiguous()开辟连续内存 view [B*L,out_size]

    assert result.size(0)==targets.size(0)

    loss=F.cross_entropy(result,targets) #Cross_entropy(x,y) x 2-D tensor shape 为 batch*n  y为大小为n的1-D tensor 包含类别索引（0- n-1)

    return loss

# CRF loss function 还未看懂

def cal_lstm_crf_loss(crf_scores,targets,tag2id):
    """计算双向LSTM-CRF模型的损失
    该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    """
    pad_id = tag2id.get('<pad>')
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')

    device = crf_scores.device

    # targets:[B, L] crf_scores:[B, L, T, T]
    batch_size, max_len = targets.size()
    target_size = len(tag2id)

    # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)

    # # 计算Golden scores方法１
    # import pdb
    # pdb.set_trace()
    targets = targets.masked_select(mask)  # [real_L]

    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()

    golden_scores = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)).sum()

    # 计算golden_scores方法２：利用pack_padded_sequence函数
    # targets[targets == end_id] = pad_id
    # scores_at_targets = torch.gather(
    #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
    # scores_at_targets, _ = pack_padded_sequence(
    #     scores_at_targets, lengths-1, batch_first=True
    # )
    # golden_scores = scores_at_targets.sum()

    # 计算all path scores
    # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                                      t, start_id, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()

    # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    loss = (all_path_scores - golden_scores) / batch_size
    return loss

#cal_lstm_crf_loss 辅助函数
def indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets



''' Helper Function for Bert '''

#sentence 用Tokenizer ， tag 需要自己增加padding
def tag_add_padding(batch,max_len,tag2id):
    PAD=tag2id.get('[PAD]')
    CLS=tag2id.get('[CLS]')
    SEP=tag2id.get('[SEP]')
    UNK=tag2id.get('[UNK]')
    batch_size=len(batch)

    batch_tensor=np.ones((batch_size,max_len))* PAD

    for i,l in enumerate(batch):
        batch_tensor[i][0]=CLS
        for j,e in enumerate(l):
            batch_tensor[i][j+1]=tag2id.get(e,UNK)
        batch_tensor[i][len(l)+1]=SEP
    
    return batch_tensor

def extend_map_bert(tag2id):
    tag2id['[UNK]']=len(tag2id)
    tag2id['[PAD]']=len(tag2id)
    tag2id['[CLS]']=len(tag2id)
    tag2id['[SEP]']=len(tag2id)
    return tag2id

if __name__=='__main__':
    pass
