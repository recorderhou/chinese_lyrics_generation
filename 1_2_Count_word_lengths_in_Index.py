import json
import numpy as np

'''
本程序根据build_word_index.py所生成的词语列表，测算出每一个index所对应的词语的字数。

'''

word_len_matrix_path="word_len_matrix.npy"

cw=json.load(open("corresponding_word.json","r",encoding="utf-8"))

word_length_matrix=np.zeros((10003))
for i in range(1,10000):
    word_length_matrix[i]=len(cw[i])
word_length_matrix[10001]=1
word_length_matrix[10002]=-1

np.save(word_len_matrix_path,word_length_matrix,allow_pickle=True)
