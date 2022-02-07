import numpy as np
import json

embedding_matrix=np.load("embedding_matrix.npy")
len_matrix=np.load("word_len_matrix.npy")
yayun_matrix=np.load("yayun_matrix.npy")

embedding_325=np.concatenate([np.expand_dims(len_matrix,1),embedding_matrix,yayun_matrix],axis=1)
# 每一个词向量，第0维是字数，1-300维是词向量，301-324维是onehot

print(embedding_325[0])
print(embedding_325[100])

np.save("embedding_325.npy",embedding_325,allow_pickle=True)