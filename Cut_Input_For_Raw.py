#coding:utf-8

import numpy as np
import json
import jieba
import re #正则表达式包

'''
这个文件包含 convert_raw_template_to_shape 和 get_index_sequence_from_raw
两个函数，以及对应的测试样例。函数详情可见程序内注释。
2020.4.30 楼涵月
'''

def convert_raw_template_to_shape(word_template):
    # word_template:一个以\n分句的中文字符的string， 形如“两只老虎 两只老虎\n跑得快\n跑得快”的目标格式。
    # 返回一个int的list，如[4,4,3,3]。
    # 还有一个分隔符的list，如[' ','\n','\n',\n']
    s=word_template+'\n'+"!"
    left=0
    right=0
    shape=[]
    puncs=[]
    while (not s[right]=='!'):
        if (s[right]=='\n' or s[right]==' '):
            puncs.append(s[right])
            sentence=s[left:right]
            left=right+1
            right=left
            uff=0
            if (sentence[0]=='\ufeff'):
                uff=1
            shape.append(len(sentence)-uff)
        else:
            right+=1
    return(shape,puncs)

def get_index_sequence_from_raw(word_index,cor_word,cut_shape,raw_text,min_take_ratio,max_take_ratio):
    # word_index: 由build_word_index.py所生成的，词→整数编号的字典文件。
    # cor_word: 由build_word_index.py所生成的，整数编号→词的list。
    # cut_shape: 形如[4,4,3,3]的目标格式。
    # raw_text: 形如“我数学很差，同学们都嘲笑我。我感到很伤心。”的原始中文字符string。
    # min/max_take_ratio: 每生成一个字抓取的字数下限和上限。如：
    # min_take_ratio=0.5,max_take_ratio=2,会为“两只老虎”在尽量保存整句的要求下抓取2-8个字。
    # 返回一个元素为list<int>的list，如[[2323,890,12],[111,532,345]]，里面的int是词的index。
    while (len(raw_text)<np.sum(cut_shape)*10):
        raw_text=raw_text*2
    strip_punc=re.split(r'，|。|“|”|：|！|？|\n|、',raw_text) # 根据标点符号分成短句
    while '' in strip_punc: # 去掉分出来的空串
        strip_punc.remove('')
    all_index=[] # 由每个短句的分词再转为index而成的list组成的list<list<int>>
    char_counts=[] # 每个短句的总合法字数
    word_counts=[] # 每个短句的总合法词数
    for words in strip_punc:
        cut_words=list(jieba.cut(words,cut_all=False)) # 一个短句分词而成的list
        cut_index=[]
        char_num=0 # 这个短句在词典内的总字数
        word_num=0 # 这个短句在词典内的总词数
        for w in cut_words:
            if w in word_index.keys(): # 在词典中
                char_num+=len(w)
                word_num+=1
                cut_index.append(word_index[w])
        all_index.append(cut_index)
        char_counts.append(char_num)
        word_counts.append(word_num)
    
    
    wholelist=[] #所有句子的list
    last_sentence_used_chars=0 # 上一次截取掉的字数
    first_unused_word=0 # 上一次截取完剩的半截子里，的第一个词的下标
    short_sen_count=len(all_index)
    sen_ptr=0 # 正在从中取词的短句的序号
    for sentence_len in cut_shape:
        thislist=[]
        already_taken_char_count=0
        min_target=sentence_len*min_take_ratio
        max_target=sentence_len*max_take_ratio
        while (already_taken_char_count+char_counts[sen_ptr]-last_sentence_used_chars<=max_target):
            # 这个短句可以全取，就全要了
            thislist+=all_index[sen_ptr][first_unused_word:] # 把整个短句剩下的词的index全拼进来
            already_taken_char_count+=char_counts[sen_ptr]-last_sentence_used_chars
            first_unused_word=0
            last_sentence_used_chars=0
            sen_ptr=(sen_ptr+1)%short_sen_count # 指针移到下一个短句
        while (already_taken_char_count<min_target):
            # 能进这个循环，说明下一块太大，一口吃不下，所以就得一个词一个词吃
            thiswordindex=all_index[sen_ptr][first_unused_word]
            thiswordlen=len(cor_word[thiswordindex])
            thislist.append(thiswordindex)
            already_taken_char_count+=thiswordlen
            last_sentence_used_chars+=thiswordlen
            first_unused_word+=1
            if (first_unused_word==len(all_index[sen_ptr])):
                # 这一步正好把这个短句吃光了
                sen_ptr=(sen_ptr+1)%short_sen_count # 指针移到下一个短句
                last_sentence_used_chars=0
                first_unused_word=0 # 又是崭新的句子了呢！
        wholelist.append(thislist)
        '''更新：剩下的半截句子不要了'''
        sen_ptr=(sen_ptr+1)%short_sen_count
        last_sentence_used_chars=0
        first_unused_word=0
    return(wholelist)


def get_some_50_words(word_index,cor_word,num_of_50s,raw_text):
    clean=""
    for char in raw_text:
        if ('\u4e00'<= char <='\u9fa5'):
            clean+=char
    word_list=list(jieba.cut(clean,cut_all=False))
    index_list=[]
    for word in word_list:
        if (word in word_index.keys()):
            index_list.append(word_index[word])
    output_list=[]
    length=len(index_list)
    if (length==0): # 如果太不走运，输入的东西都不在词表里，那就加个啊吧……
        index_list.append(word_index['啊'])
        length=1
    ptr=0
    for i in range(num_of_50s):
        out=[]
        for j in range(50):
            out.append(index_list[ptr])
            ptr=(ptr+1)%length
        output_list.append(out)
    return(output_list)

def get_index_sequence_from_text(word_index,raw_text):
    output=[]
    cut_words=list(jieba.cut(raw_text,cut_all=False)) # 一个短句分词而成的list
    for word in cut_words:
        if (word in word_index.keys()):
            output.append(word_index[word])
    return output

'''
函数测试：


w_index=json.load(open("word_index.json","r",encoding="utf-8"))
c_word=json.load(open("corresponding_word.json","r",encoding="utf-8"))

raw_template="两只老虎 两只老虎\n跑得快\n跑得快\n一只不会微分\n一只不会积分\n真奇怪\n真奇怪"

shape,p=convert_raw_template_to_shape(raw_template)
print(shape)
print(p)

raw_text="我数学很差，同学们都嘲笑我，说“你学不会多元微分的”。我感到很伤心。"
index_sequence=get_index_sequence_from_raw(w_index,c_word,shape,raw_text,0.5,2.0)
print(index_sequence)

for i in range(len(index_sequence)):
    for j in range(len(index_sequence[i])):
        index_sequence[i][j]=c_word[index_sequence[i][j]]
print(index_sequence)

test_50 = get_some_50_words(w_index,c_word,3,"你好啊我的名字叫楼涵月！我喜欢写程序！这是第155行！但我不喜欢bug！")
print(test_50)
'''