import numpy as np
import gensim
import json

'''封装数据集：将数据集向量化并返回batch'''
class Creator:

    def __init__(self, input_texts, target_texts,encoder_length,decoder_length ):
        self.input_texts = np.asarray(input_texts,dtype = np.int32)
        self.target_texts = np.asarray(target_texts,dtype = np.int32)
        self.encoder_length = np.asarray(encoder_length,dtype = np.int32)
        self.decoder_length = np.asarray(decoder_length,dtype = np.int32)
        self._indicator = 0

        self.random_shuffle() # 对数据进行随机化

    def random_shuffle(self): # 数据随机化
        p = np.random.permutation(len(self.encoder_length))
        self.input_texts = self.input_texts[p]
        self.target_texts = self.target_texts[p]
        self.encoder_length = self.encoder_length[p]
        self.decoder_length = self.decoder_length[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size # batch结束的位置
        if end_indicator > len(self.target_texts):
            self.random_shuffle()
            self._indicator = 0
            end_indicator = batch_size # 如果结束的位置超出范围，就把开始位置移到开头，并且把数据进行一次随机化
        if end_indicator > len(self.target_texts):
            raise Exception('batch_size: %d is too large' % batch_size) # 避免batch_size比所有数据之和还大的情况
        input_texts_batch = self.input_texts[self._indicator: end_indicator]
        target_texts_batch = self.target_texts[self._indicator: end_indicator]
        encoder_length_batch = self.encoder_length[self._indicator: end_indicator]
        decoder_length_batch = self.decoder_length[self._indicator: end_indicator]
        self._indicator = end_indicator # 迭代，下一批的开头是这一批的末尾
        return input_texts_batch,target_texts_batch,encoder_length_batch,decoder_length_batch