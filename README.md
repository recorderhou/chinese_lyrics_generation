# chinese_lyrics_generation
### 奉码填词：AI歌词生成
### 作者： 
### 时间：2020/5/12 

【奉码填词：AI歌词生成】是一个输入歌词模板（即歌中每一句的长度）与生成素材（任意题材的任意长度文本）、输出填好歌词的模型。以下是正常运行【奉码填词】程序所需要完成的步骤。

0：下载原始词向量数据。见"0-原始词向量数据.txt"。它的名字是"sgns.literature.bigram-char"。

1：依次运行以下数据处理文件。
"1_0_Clean_build_word_index.py"：这个程序根据原始词向量建立词表，并生成：
>"embedding_matrix.npy"
>"word_index.json"
>"corresponding_word_path.json"

"1_1_Make_Pinyin_Matrix.py"：这个程序利用插件pypinyin求出词表中每一个词韵脚的24维onehot表示，并生成：
>"yayun_matrix.npy"

"1_2_Count_word_lengths_in_Index.py"：这个程序生成记录每个词长度的：
>"word_len_matrix.npy"

"1_3_Build_325_embedding.py"：这个程序把上述三个numpy矩阵拼成：
>"embedding_325.npy"

2：将"2_2_UI素材.rar"解压。里面是一些图片素材和预设好的歌词模板。

3：以下是一些不需要操作的文件：
"Cut_Input_For_Raw.py" 里是几个基于jieba插件对输入模板、素材进行预处理的函数。
"Final_Deep_Model.py"里是一个基于Attention机制的seq2seq生成模型。
"Final_Shallow_Model.py"里是一个基于LSTM评价函数的生成模型。
"generation_model"：这个文件夹里应该是Final_Deep_Model的模型参数，总大小约200MB，可以向我们索取。
"models"：这个文件夹里应该是Final_Shallow_Model的模型参数，总大小约14MB，可以向我们索取。

4：以上步骤都完成之后，【奉码填词】就可以运行啦！
>运行"奉码填词_浅.py"，使用基于"Final_Shallow_Model.py"的歌词生成器。
>运行"奉码填词_深.py"，使用基于"Final_Deep_Model.py"的歌词生成器。

祝您体验愉快！
