# -*- coding: utf-8 -*-

# 把"词语 词向量"格式的百度百科词向量txt，洗成只有1w*300个浮点数的embedding矩阵。
# 每一行代表哪个词，可以通过corresponding_word字典来查询。

# 然后，建立词语和索引之间的关系。
# 对于每一个词语 "这个词"，word_index["这个词"] 是这个词的索引。
# 这意味着：
# 在词向量矩阵embedding中，embedding[word_index["这个词"]]是"这个词"的词向量；
# 在韵母矩阵vowel中，vowel[word_index["这个词"]]是"这个词"的韵母向量。
# 对于每一个整数 i ,corresponding_word[i]代表着embedding、vowel两个矩阵第i行所对应的中文词语。
# 楼涵月 2020/4/23
'''
# 这个程序输入来自github的百度百科bigram-char文本，输出三个文件。
# embedding_matrix.npy 保存的是一个numpy矩阵，里面每一行代表一个词向量。
# word_index.npy 保存的是一个字典，其中键是词语，值是整数。
# corresponding_word.npy 保存的是一个列表，其中第i个元素是整数i对应的词语。
'''
import numpy as np
import json

embedding_source="sgns.literature.bigram-char" #下载下来的原始词向量的路径
embedding_path="embedding_matrix.npy" #输出的矩阵路径
word_index_path="word_index.json" #输出的词→数字典路径
corresponding_word_path="corresponding_word.json" #输出的数→词词典路径

total_word_count=0

word_index={"":0}
corresponding_word=[""]
with open(embedding_source,"r",encoding="utf-8") as source:
    temp=source.readline().strip("\n").split(" ")
    total_line_count=int(temp[0])
    total_word_count = 1 # 因为要洗掉含非中文字符的词语，所以要重新来过
    embedding_matrix = np.zeros((10003,300)) # 一共187980 个词
    for i in range(total_line_count):
        temp = source.readline().strip("\n").split(" ")
        word = temp[0]
        good = 1
        for ch in word:
            if (not ('\u4e00'<= ch <='\u9fa5')):
                good=0 # 去掉有非中文字符的内容
        if (good):            
            word_index[temp[0]]=total_word_count
            corresponding_word.append(temp[0])
            for k in range(300):
                embedding_matrix[total_word_count][k]=float(temp[k+1])
            total_word_count += 1
        if (total_word_count==10000):
            break
print(total_word_count) # 最后一共181339个词

corresponding_word.append('<PAD>')
word_index['<PAD>']=10000
corresponding_word.append('<EOS>')
word_index['<EOS>']=10001
for i in range(300):
    embedding_matrix[10001]=1
corresponding_word.append('<GO>')
word_index['<GO>']=10002
for i in range(300):
    embedding_matrix[10002]=-1

np.save(embedding_path,embedding_matrix,allow_pickle=True)

with open(word_index_path,"w",encoding='utf-8') as outfile:
    json.dump(word_index,outfile,ensure_ascii=False)
with open(corresponding_word_path,"w",encoding='utf-8') as outfile:
    json.dump(corresponding_word,outfile,ensure_ascii=False)