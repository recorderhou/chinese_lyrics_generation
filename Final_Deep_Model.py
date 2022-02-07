import tensorflow as tf
import json
from Batch_Creator import Creator
from tensorflow.python.util import nest
import os
import numpy as np
import re
import jieba
from Cut_Input_For_Raw import get_some_50_words
import math

embedding_325 = np.load('embedding_325.npy')
word_index = json.load(open("word_index.json", "r", encoding='utf-8'))
cor_word = json.load(open("corresponding_word.json", "r", encoding='utf-8'))

target_texts = json.load(open("train_target_texts","r",encoding='utf-8'))
input_texts = json.load(open("train_input_texts","r",encoding='utf-8'))
encoder_sequence_length_all = json.load(open("train_encoder_sequence_length_all","r",encoding='utf-8'))
decoder_sequence_length_all = json.load(open("train_decoder_sequence_length_all","r",encoding='utf-8'))

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


def input_to_nums():
    input_nums = []
    target_nums = []
    for sentence in input_texts:
        tmp = []
        for words in sentence:
            tmp.append(word_index[words])
        input_nums.append(tmp)
    for sentence in target_texts:
        tmp = []
        for words in sentence:
            tmp.append(word_index[words])
        target_nums.append(tmp)
    return input_nums, target_nums

def turn_passage_to_verse(path):
    fd = open(path, 'rb')
    ret = fd.read().decode('utf-8', 'ignore')
    fifs_needed = len(ret) // 500
    word_seq = get_some_50_words(word_index, cor_word, fifs_needed, ret)
    return word_seq, fifs_needed

def turn_string_to_verse(ret):
    fifs_needed = 5 # len(ret) // 500
    word_seq = get_some_50_words(word_index, cor_word, fifs_needed, ret)
    return word_seq, fifs_needed

def turn_predicted_to_words(predicted_ids, cor_word, beam_size):
    for prediction in predicted_ids:
        for i in range(beam_size):
            predict_list = np.ndarray.tolist(prediction[:, :, i])
            predict_seq = [cor_word[idx] for idx in predict_list[0]]
            return("".join(predict_seq))

class Seq2SeqModel():
    def __init__(self,sess, rnn_size, num_layers, embedding_size, learning_rate, embedding,
                 vocab_size, word_to_idx, mode, use_attention,
                 beam_search, beam_size, max_gradient_norm=5.0):
        self.learning_rate = learning_rate  # 学习率
        self.embedding_size = embedding_size  # 词向量维数
        self.rnn_size = rnn_size  # 隐层神经元数
        self.num_layers = num_layers  # 神经元层数
        self.embedding = embedding  # embedding向量，[vocab_size, embedding_size]
        self.vocab_size = vocab_size  # 这是歌名和歌词的词表大小
        self.word_to_idx = word_to_idx  # word_index
        self.mode = mode  # 模式，是train和decoder
        self.use_attention = use_attention  # 是否使用attention机制
        self.beam_search = beam_search  # 是否使用beam_search
        self.beam_size = beam_size  # beam_size 是每次搜索选取前beam_size个可能结果
        self.max_gradient_norm = max_gradient_norm  # 梯度裁剪参数
        # 执行模型构建部分的代码
        self.build_model(sess)



    def _create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            # 添加dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell

        # 列表中每个元素都是调用single_rnn_cell函数
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self,sess):
        print('building model... ...')
        # 定义模型的placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, 50], name='encoder_inputs')  # 输入序列
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')  # 输入序列的长度
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # batch 大小
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')  # 遗忘率
        self.decoder_targets = tf.placeholder(tf.int32, [None, 10], name='decoder_targets')  # 想要decoder生成的目标序列
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')  # 目标序列长度
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')  # 最长目标序列
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32,
                                     name='masks')  # mask层，用于在计算时忽略padding
        self.sampling_probability = tf.placeholder(tf.float32, [], name='sampling_probability')  # teacher-forcing需要


        # encoder部分
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # 创建LSTMCell，两层+dropout
            encoder_cell = self._create_rnn_cell()
            #  使用预训练的词向量作为embedding层，歌名和歌词共享一个词表
            embedding = self.embedding
            embedding = tf.cast(embedding, tf.float32)
            encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            encoder_inputs_embedded = tf.cast(encoder_inputs_embedded, tf.float32)

            # 使用dynamic_rnn构建LSTM模型，将输入编码成隐层向量。
            # encoder_outputs用于attention，batch_size * encoder_inputs_length*rnn_size,
            # encoder_state用于decoder的初始化状态，batch_size * rnn_szie
            #  input : [batch_size, max_steps, n_input(向量维数)]
            #  output : [batch_size, max_time, hidden_size]
            #  state : tuple (num_layers, [2, batch_size, hidden_size])
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)

        # decoder部分
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            encoder_inputs_length = self.encoder_inputs_length
            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                #  这里因为要用到beam_search，所以进行一次tile
                #  input : [batch_size, max_step, hidden_size]
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
                #  output : [batch_size * beam_size, max_step, hidden_size]

                #  input : tuple (num_layers, [2(分别是c和h), batch_size, hidden_size])
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size),
                                                   encoder_state)
                #  output : 也是一样复制了beam_size份

                #  input : [batch_size]
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length,
                                                                           multiplier=self.beam_size)
                #  output : [batch_size * beam_size]

            #  加入attention
            #  input : [batch_size / 12, max_step, hidden_size]
            self.local_nums = 12
            attention_feed_outputs = tf.split(encoder_outputs, self.local_nums, -1)
            attention_feed_output = attention_feed_outputs[-1]
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size,
                                                                    memory=attention_feed_output,
                                                                    memory_sequence_length=encoder_inputs_length)

            # 定义decoder阶段要是用的LSTMCell，然后为其加上attention wrapper
            decoder_cell = self._create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size,
                                                               name='Attention_Wrapper')

            # 使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            # 定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
                cell_state=encoder_state)

            # 需要添加一个全连接层
            output_layer = tf.layers.Dense(self.vocab_size,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if self.mode == 'train':
                # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
                # decoder_inputs_embedded的shape为[batch_size, decoder_targets_length, embedding_size]
                #  decoder_target即处理好的input_target，因为用到teacher forcing 所以需要处理输入
                #  去掉<EOS>，加上<GO>

                ending = tf.strided_slice(self.decoder_targets, [0, 0], [batch_size, -1], [1, 1])
                print(ending.eval())
                #  在 1 维度拼接，也就是在词表前加上<GO>对应的下标，由于有batch_size个target，保证维数一样进行拼接
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<GO>']), ending], 1)
                print(decoder_input.eval())
                #  事实上如果使用预训练的词向量，只需要把每个batch里<EOS>对应的向量删去，再加上<GO>对应的词向量就好
                #  我们的embedding就是预训练的词向量
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
                decoder_inputs_embedded = tf.cast(decoder_inputs_embedded, tf.float32)

                # 训练阶段
                training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=decoder_inputs_embedded,
                                                                       sequence_length=self.decoder_targets_length,
                                                                       embedding=embedding,
                                                                       sampling_probability=self.sampling_probability,
                                                                       time_major=False,
                                                                       seed=None,
                                                                       scheduling_seed=None,
                                                                       name='training_helper')

                #  decoder_targets_length = [self.max_target_sequence_length] * batch_size
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                                   initial_state=decoder_initial_state,
                                                                   output_layer=output_layer)

                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案

                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=
                                                                          self.max_target_sequence_length)

                # 根据输出计算loss和梯度，并定义进行更新的Optimizer和train_op
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                #  labels的维度是[batch_size * max_seq_size], 而logits的维度为[可变seq_size * batch_size, vocab_size],
                #  这样就会导致矩阵运算无法运行，因此暴力地把他们切得一样长
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                current_ts = tf.to_int32(tf.minimum(tf.shape(self.decoder_targets)[1], tf.shape(self.decoder_logits_train)[1]))
                decoder_targets = tf.slice(self.decoder_targets, begin=[0, 0], size=[-1, current_ts])
                mask = tf.sequence_mask(self.decoder_targets_length, current_ts,
                                        dtype=tf.float32,
                                        name='mask')
                # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
                decoder_logits_train = tf.slice(self.decoder_logits_train, begin=[0, 0, 0], size=[-1, current_ts, -1])
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits_train,
                                                             targets=decoder_targets, weights=mask)

                #  tf.summary.scalar('loss', self.loss)
                #  self.summary_op = tf.summary.merge_all()
                #  cross_entropy = -tf.reduce_mean(y_input * tf.log(y_pre))
                self.train_op = tf.train.RMSPropOptimizer(self.learning_rate,
                                           momentum=0.9, decay=0.9).minimize(self.loss)
                '''self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.loss)'''
                '''trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))'''

            elif self.mode == 'decode':

                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<GO>']
                end_token = self.word_to_idx['<EOS>']

                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=embedding,
                                                                             start_tokens=start_tokens,
                                                                             end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)

                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                          maximum_iterations=10)

                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
                # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                # beam_search_decoder_output:
                # BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
                if self.beam_search:
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)  # 其实用不着

        # =================================4, 保存模型
        #sess.run(tf.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        self.saver.restore(sess,tf.train.latest_checkpoint('generation_model'))

    def train(self, sess, batch):
        # 对于训练阶段，需要执行self.train_op, self.loss三个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob_placeholder: 0.5,
                     self.batch_size: len(batch.encoder_inputs),
                     self.sampling_probability: 0.1}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob_placeholder: 1.0,
                     self.batch_size: len(batch.encoder_inputs),
                     self.sampling_probability: 0.1}

        loss = sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def infer(self, sess, batch):
        # infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.keep_prob_placeholder: 1.0,
                     self.batch_size: len(batch.encoder_inputs),
                     self.sampling_probability: 0.1}
        print(self.encoder_inputs)
        print(self.encoder_inputs_length)
        print(batch.encoder_inputs_length)
        sess.run(tf.global_variables_initializer())
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict


class batch():
    def __init__(self, input_texts, target_texts, encoder_length, decoder_length):
        self.encoder_inputs = input_texts
        self.encoder_inputs_length = encoder_length
        self.decoder_targets = target_texts
        self.decoder_targets_length = decoder_length
        print(encoder_length)
        print(self.encoder_inputs_length)

def generate_lyrics_attention(source):
    output=""
    with tf.Session() as sess:
        model = Seq2SeqModel(sess, 600, 4, 325, 0.0001, embedding_325, len(word_index.keys()), word_index,
                            mode='decode', use_attention=True, beam_search=True, beam_size=5, max_gradient_norm=5.0)
        #ckpt = tf.train.get_checkpoint_state('generation_models')
        #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess,tf.train.latest_checkpoint('generation_model'))
        #  这里predict的部分batch_size = 1
        #  读取文件↓
        input_sequence, fifs_needed = turn_string_to_verse(source)
        sequence_length = [50]
        for sentences in input_sequence:
            tmp_list = []
            tmp_list.append(sentences)
            pred_batch = batch(tmp_list, [[0]], [50], [[0]])
            # 获得预测的id
            predicted_ids = model.infer(sess, pred_batch)
            output+=turn_predicted_to_words(predicted_ids, cor_word, 5)
            output+="\n"
        return(output)

#  o=generate_lyrics_attention("看官你道此书从何而起?说来虽近荒唐，细玩颇有趣味。却说那女娲氏炼石补天之时，于大荒山无稽崖炼成高十二丈、见方二十四丈大的顽石三万六千五百零一块。那娲皇只用了三万六千五百块，单单剩下一块未用，弃在青埂峰下。谁知此石自经锻炼之后，灵性已通，自去自来，可大可小。因见众石俱得补天，独自己无才不得入选，遂自怨自愧，日夜悲哀。一日正当嗟悼之际，俄见一僧一道远远而来，生得骨格不凡，丰神迥异，来到这青埂峰下，席地坐谈。见着这块鲜莹明洁的石头，且又缩成扇坠一般，甚属可爱。那僧托于掌上，笑道：“形体倒也是个灵物了，只是没有实在的好处。须得再镌上几个字，使人人见了便知你是件奇物，然后携你到那昌明隆盛之邦、诗礼簪缨之族、花柳繁华地、温柔富贵乡那里去走一遭。”")
#  print(o)

if __name__ == '__main__':
    total_loss = 0.0
    with tf.Session() as sess:
        input_nums, target_nums = input_to_nums()
        creator = Creator(input_nums, target_nums, encoder_sequence_length_all, decoder_sequence_length_all)
        model = Seq2SeqModel(sess, 600, 4, 325, 0.001, embedding_325, len(word_index.keys()), word_index, mode='train',
                            use_attention=True,
                            beam_search=False,
                            beam_size=5,
                            max_gradient_norm=5.0
                            )
        current_step = 0
        for e in range(5000):
            print("----- Epoch {}/{} -----".format(e + 1, 5000))
            en_input, de_input, en_seq_len, de_seq_len = creator.next_batch(16)
            real_batch = batch(en_input, de_input, en_seq_len, de_seq_len)
            loss = model.train(sess, real_batch)
            total_loss += loss
            current_step += 1
            # 每多少步进行一次保存
            if current_step % 200 == 0:
                print("----- Step %d -- Loss %.2f" % (current_step, total_loss/current_step))
                #  summary_writer.add_summary(summary, current_step)
                model.saver.save(sess, "generation_model/model", global_step=current_step)
                o=generate_lyrics_attention("看官你道此书从何而起?说来虽近荒唐，细玩颇有趣味。却说那女娲氏炼石补天之时，于大荒山无稽崖炼成高十二丈、见方二十四丈大的顽石三万六千五百零一块。那娲皇只用了三万六千五百块，单单剩下一块未用，弃在青埂峰下。谁知此石自经锻炼之后，灵性已通，自去自来，可大可小。因见众石俱得补天，独自己无才不得入选，遂自怨自愧，日夜悲哀。一日正当嗟悼之际，俄见一僧一道远远而来，生得骨格不凡，丰神迥异，来到这青埂峰下，席地坐谈。见着这块鲜莹明洁的石头，且又缩成扇坠一般，甚属可爱。那僧托于掌上，笑道：“形体倒也是个灵物了，只是没有实在的好处。须得再镌上几个字，使人人见了便知你是件奇物，然后携你到那昌明隆盛之邦、诗礼簪缨之族、花柳繁华地、温柔富贵乡那里去走一遭。”")
                print(o)

    o=generate_lyrics_attention("看官你道此书从何而起?说来虽近荒唐，细玩颇有趣味。却说那女娲氏炼石补天之时，于大荒山无稽崖炼成高十二丈、见方二十四丈大的顽石三万六千五百零一块。那娲皇只用了三万六千五百块，单单剩下一块未用，弃在青埂峰下。谁知此石自经锻炼之后，灵性已通，自去自来，可大可小。因见众石俱得补天，独自己无才不得入选，遂自怨自愧，日夜悲哀。一日正当嗟悼之际，俄见一僧一道远远而来，生得骨格不凡，丰神迥异，来到这青埂峰下，席地坐谈。见着这块鲜莹明洁的石头，且又缩成扇坠一般，甚属可爱。那僧托于掌上，笑道：“形体倒也是个灵物了，只是没有实在的好处。须得再镌上几个字，使人人见了便知你是件奇物，然后携你到那昌明隆盛之邦、诗礼簪缨之族、花柳繁华地、温柔富贵乡那里去走一遭。”")
    print(o)
