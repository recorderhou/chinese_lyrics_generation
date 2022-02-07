# coding-utf-8

import numpy as np
import json
from pypinyin import pinyin, lazy_pinyin, Style
verse_list = np.array(["ong", "ang", "hi", "ei", "v", "i", "ia", "ui", "en", "uan", "an", "ian", "ao", "e", "a", "eng",
              "ing", "ou", "in", "ie", "ai", "u", "o", "r"])

def verse_to_mat(word):
    word_mat = np.array([0] * 24)
    verse_store = pinyin(word, style=Style.TONE3, heteronym=False)
    for i in range(0, 24):
        if verse_store[-1][-1][-1].isdigit():
            test_str = verse_store[-1][-1][-2-len(verse_list[i]):-1]
        else:
            test_str = verse_store[-1][-1][-1-len(verse_list[i]):]
        if test_str[1:] == verse_list[i]:
            # 一些特判
            if verse_list[i] == "an":
                if test_str[0] == "y":
                    word_mat[i + 1] = 1
                elif verse_store[0] != "u":
                    word_mat[i] = 1
                else:
                    word_mat[i - 1] = 1

            elif verse_list[i] == "e":
                if test_str[0] == "y":
                    word_mat[19] = 1
                elif test_str[0] == "i":
                    word_mat[19] = 1
                else:
                    word_mat[i] = 1

            elif verse_list[i] == "u":
                if test_str[0] == "y":
                    word_mat[4] = 1
                elif test_str[0] == "q":
                    word_mat[4] = 1
                elif test_str[0] == "j":
                    word_mat[4] = 1
                elif test_str[0] == "x":
                    word_mat[4] = 1
                else:
                    word_mat[i] = 1

            elif verse_list[i] == "i":
                if test_str[0] == "h":
                    word_mat[2] = 1
                else:
                    word_mat[i] = 1

            elif verse_list[i] == "r":
                if test_str[0] == "e":
                    word_mat[23] = 1

            else:
                word_mat[i] = 1
    return word_mat

# print(pinyin("中心", style=Style.TONE3, heteronym=False))

cor_word=json.load(open("corresponding_word.json","r",encoding="utf-8"))

word_count=len(cor_word)-3
print(word_count)
yayun_matrix=[]
for i in range(word_count):
    if (i==0):
        yayun_matrix.append(np.ones((24)))
    else:
        yayun_matrix.append(verse_to_mat(cor_word[i]))
yayun_matrix.append(np.zeros(24))
yayun_matrix.append(np.ones(24))
temp=np.zeros(24)
for i in range (24):
    temp[i]=-1
yayun_matrix.append(temp)
yayun_matrix=np.array(yayun_matrix)
np.save("yayun_matrix.npy",yayun_matrix,allow_pickle=True)
    