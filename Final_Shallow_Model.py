#coding:utf-8

import numpy as np
import json
import tensorflow as tf

from Cut_Input_For_Raw import convert_raw_template_to_shape
from Cut_Input_For_Raw import get_index_sequence_from_raw

embedding_matrix=np.load("embedding_matrix.npy")[0:10000,:]
len_matrix=(np.load("word_len_matrix.npy"))[0:10000]
word_index=json.load(open("word_index.json","r",encoding="utf-8"))
cor_word=json.load(open("corresponding_word.json","r",encoding="utf-8"))
yayun_matrix=np.load("yayun_matrix.npy")[0:10000,:]
embedding_325=np.load("embedding_325.npy")[0:10000,:]

embed_dimension=300 # 词向量的维度，是常量
dict_length=10000 # 字典的词数
hidden_size = 256    # 隐含层的大小
class_num=2 
batch_size = 1     # 批量大小（因为每一句的词数都不相等，所以就得一句一句来）
layer_num = 2        # LSTM layer 的层数
timestep_size = 10
lr=0.001 #学习率
cell_type = 'lstm'   # lstm 或者 block_lstm
model_dir = 'judgment_models' # 存储路径

#转换为tensor
embedding_tensor = tf.convert_to_tensor(embedding_matrix,dtype=tf.float32)
len_tensor=tf.convert_to_tensor(len_matrix)
yayun_tensor=tf.convert_to_tensor(yayun_matrix,dtype=tf.int64)
embedding_325_tensor=tf.convert_to_tensor(embedding_325,dtype=tf.float32)

source_input = tf.placeholder(tf.int64, [None]) # 一维向量（一次只变形一句），素材句里每一个词的编号
input_word_count = tf.placeholder(tf.int64,[]) # 句子里的词总数
output_char_count = tf.placeholder(tf.int64,[]) # 要输出的字数
yayun_vector=tf.placeholder(tf.int64,[24])
keep_prob = tf.placeholder(tf.float32, []) # LSTM单元的keep_prob

emb_input_= tf.nn.embedding_lookup(embedding_tensor, source_input) # [input_word_count,300] 的embed后矩阵
random_cover=tf.random_normal((input_word_count,300),dtype=tf.float32,stddev=1) # 生成一点随机噪音
emb_input=emb_input_+random_cover

# 创建 lstm 结构
def lstm_cell(cell_type, num_nodes, keep_prob):
    if cell_type == 'lstm':
        cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
    else:
        cell = tf.contrib.rnn.LSTMBlockCell(num_nodes)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell



def total_wordlen(source_inputs): #输出总字数
    lens=tf.nn.embedding_lookup(len_tensor,source_inputs)
    len_sum=tf.cast(tf.reduce_sum(lens),dtype=tf.int64)
    return(len_sum)

def change_one_position_1D(original_vector,original_len,change_index,target_num):
    onehot=tf.one_hot(change_index,tf.cast(original_len,dtype=tf.int32),dtype=tf.float32)
    float_vector=tf.cast(original_vector,dtype=tf.float32)
    new_vector=float_vector+tf.multiply(tf.cast((target_num-original_vector[change_index]),dtype=tf.float32),onehot)
    int_vector=tf.cast(new_vector,dtype=tf.int64)
    return(int_vector)



def chars_less_than_target(input_word_count,source_inputs,target_length,i):
    total_chars=total_wordlen(source_inputs)
    return(tf.less(total_chars,target_length)&tf.less(i,30)) 
def chars_more_than_target(input_word_count,source_inputs,target_length,i):
    total_chars=total_wordlen(source_inputs)
    return(tf.greater(total_chars,target_length)&tf.less(i,30))

def reduce_one_char(input_word_count,source_inputs,target_length,i):
    emb_matrix=tf.nn.embedding_lookup(embedding_tensor,source_inputs)
    simil=tf.matmul(embedding_tensor,tf.transpose(emb_matrix))
    # 把现有词和所有词的长度沿不同维度扩展，算出一个50w*input_word_count矩阵，
    # 其中第i,j个元素是词表第i个词的字数与句子第j个词的字数之差*(-1)
    my_len=tf.tile(tf.expand_dims(tf.nn.embedding_lookup(len_tensor,source_inputs),0),[dict_length,1])
    whole_len=tf.tile(tf.transpose(tf.expand_dims(len_tensor,0)),[1,input_word_count])
    len_diff=tf.cast(my_len-whole_len,dtype=tf.float32)-0.99
    # 在len_diff里，比句子中的词少1个字的词是0.01，少2个字的是1.01；多1个字的是-1.99
    ones=tf.ones_like(len_diff)
    len_score=tf.cast(tf.divide(ones,len_diff),dtype=tf.float32)
    # 在len_score里，少1个字的替换得分100，少2个字的得分1，多字的得分为负数
    whole_score=tf.add(simil,len_score)
    max_indexes=tf.argmax(whole_score,0) # whole_score矩阵中，每一列的最大值所在的行序号
    flat_score=tf.reduce_max(whole_score,0) # 压扁的矩阵。whole_score矩阵中，每一列的最大值
    max_col=tf.argmax(flat_score) # whole_score全局最大值所在的列号
    max_row=max_indexes[max_col] # whole_score全局最大值所在的行号
    new_input=change_one_position_1D(source_inputs,input_word_count,max_col,max_row)
    return (input_word_count,new_input,target_length,i+1)

def add_one_char(input_word_count,source_inputs,target_length,i):
    emb_matrix=tf.nn.embedding_lookup(embedding_tensor,source_inputs)
    simil=tf.matmul(embedding_tensor,tf.transpose(emb_matrix))
    # 把现有词和所有词的长度沿不同维度扩展，算出一个50w*input_word_count矩阵，
    # 其中第i,j个元素是词表第i个词的字数与句子第j个词的字数之差
    my_len=tf.tile(tf.expand_dims(tf.nn.embedding_lookup(len_tensor,source_inputs),0),[dict_length,1])
    whole_len=tf.tile(tf.transpose(tf.expand_dims(len_tensor,0)),[1,input_word_count])
    len_diff=tf.cast(whole_len-my_len,dtype=tf.float32)-0.99
    # 在len_diff里，比句子中的词多1个字的词是0.01，多2个字的是1.01；少1个字的是-1.99
    ones=tf.ones_like(len_diff)
    len_score=tf.cast(tf.divide(ones,len_diff),dtype=tf.float32)
    # 在len_score里，多1个字的替换得分100，多2个字的得分1，少字的得分为负数
    whole_score=tf.add(simil,len_score)
    max_indexes=tf.argmax(whole_score,0) # whole_score矩阵中，每一列的最大值所在的行序号
    flat_score=tf.reduce_max(whole_score,0) # 压扁的矩阵。whole_score矩阵中，每一列的最大值
    max_col=tf.argmax(flat_score) # whole_score全局最大值所在的列号
    max_row=max_indexes[max_col] # whole_score全局最大值所在的行号
    new_input=change_one_position_1D(source_inputs,input_word_count,max_col,max_row)
    return (input_word_count,new_input,target_length,i+1)


def generate_output(input_word_count,source_inputs,emb_inputs,target_length,last_yayun_vector,previous_outputs):
    # source_inputs: 原始数据的index们
    # emb_inputs: input_word_count*300 的tensor
    # target_length: 目标输出字数
    # last_yayun_vector: 上一个押的韵, 1*24向量
    # previous_outputs: 代表之前所有句子的向量合成的矩阵

    last_word=tf.slice(emb_inputs,[input_word_count-11,0],[1,embed_dimension])
    last_similarity=tf.matmul(embedding_tensor,tf.transpose(last_word))
    # 50w*1 的相似度
    last_yayun=tf.matmul(yayun_tensor,tf.expand_dims(tf.transpose(last_yayun_vector),1))
    # 50w*1
    last_change_scores=last_similarity+100*tf.cast(last_yayun,dtype=tf.float32)
    change_to=tf.argmax(last_change_scores,0) #评分最好的词的index
    source_inputs_1=change_one_position_1D(source_inputs,input_word_count,input_word_count-1,change_to)
    i=0
    input_word_count,source_inputs_1,target_length,i=\
        tf.while_loop(chars_less_than_target,add_one_char,[input_word_count,source_inputs_1,target_length,i])
    #只要不够就补
    input_word_count,source_inputs_1,target_length,i=\
        tf.while_loop(chars_more_than_target,reduce_one_char,[input_word_count,source_inputs_1,target_length,i])
    #只要多了就减
    return(tf.cast(source_inputs_1,dtype=tf.int64))

out=generate_output(input_word_count,source_input,emb_input,output_char_count,yayun_vector,tf.zeros((300,1)))
wc=total_wordlen(out)

emb_out=tf.expand_dims(tf.nn.embedding_lookup(embedding_325_tensor,out),axis=0)
mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(cell_type, hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple = True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化state
state = init_state
# 按时间步展开计算
outputs = list()
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        (cell_output, state) = mlstm_cell(emb_out[:,timestep, :],state)
        outputs.append(cell_output)
h_state = outputs[-1] # 最后一层的输出

# 接softmax层以进行分类
W = tf.Variable(tf.zeros([hidden_size, class_num], dtype=tf.float32), dtype=tf.float32)
bias = tf.Variable(tf.zeros(shape=[class_num],dtype=tf.float32), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
score=100*(y_pre[0][1])



saver=tf.train.Saver()
def shallow_generate_lyrics(raw_template,raw_source): # 用测试集检验lstm模型训练成果
    with tf.compat.v1.Session() as sess:
        saver.restore(sess,'models/model-ckpt')
        template,puncs=convert_raw_template_to_shape(raw_template)
        index_sequence=get_index_sequence_from_raw(word_index,cor_word,template,raw_source,0.25,1.2)

        best_best_result=[]
        best_best_score=-1e6
        for yun in range(24):
            total_score=0
            total_output=[]
            last_yayun_v=np.zeros((24))
            last_yayun_v[yun]=1
            repeat_dict={}            
            for i in range(len(index_sequence)):
                i_word_count=len(index_sequence[i])
                best_out=[]
                best_score=-1e6
                for turn in range(10):
                    out_np,score_= sess.run([out,score],feed_dict={\
                        source_input:index_sequence[i]+[0]*10,\
                        input_word_count:i_word_count+10,\
                        output_char_count:template[i],\
                        yayun_vector:last_yayun_v,\
                        keep_prob:0.5})
                    yayun_v=yayun_matrix[out_np[-1]]

                    '''
                    output=""
                    for word in out_np:
                        output+=cor_word[word]
                    print(output+" Score: "+str(score_))                    
                    '''

                    score_+=50*(np.sum(yayun_v*last_yayun_v))

                    if (score_>best_score):
                        best_score=score_
                        best_out=out_np
                last_yayun_v=yayun_matrix[best_out[-1]] # 更新尾韵
                total_score+=best_score
                total_output.append(best_out)
                repeat_dict[best_out[-1]]=repeat_dict.get(best_out[-1],0)+1
            total_score-=20*max(repeat_dict.values()) # 对尾韵重复次数过多的版本进行惩罚
            if (total_score>best_best_score):
                best_best_score=total_score
                best_best_result=total_output
        
        display=""
        for i in range(len(best_best_result)):
            for index in best_best_result[i]:
                display+=cor_word[index]
            display+=puncs[i]
        return(display)
    
                    


'''
with open("Test_Raw_Template.txt","r",encoding="utf-8") as tem:
    raw_template=tem.read()
with open("Test_Raw_Source.txt","r",encoding="utf-8") as rf:
    raw_source=rf.read()
result=shallow_generate_lyrics(raw_template,raw_source)
print(result)
'''