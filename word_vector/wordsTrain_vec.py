# author:fighting
# from langconv import *
import os
from gensim.models.word2vec import Word2Vec
# def Simplified2Traditional(sentence):
#     '''
#     将sentence中的简体字转为繁体字
#     :param sentence: 待转换的句子
#     :return: 将句子中简体字转换为繁体字之后的句子
#     '''
#     sentence = Converter('zh-hans').convert(sentence)
#     return sentence

# LCQMC dataset
dirname = './data/LCQMC'
sentence = []
words = []
for filename in os.listdir(dirname):
    with open(os.path.join(dirname, filename), 'r') as lcqmc:
        for line in lcqmc:
            linedict = eval(line)
            word = linedict['sentence1']
            poss = linedict['sentence2']
            sentence.append(word)
            sentence.append(poss)

with open('../data/public_simple_data/data_text', 'r') as f:
    for line in f:
        # line = Simplified2Traditional(line)

        sentence.append(line)
print("data_text size:", len(sentence))
# sighan 2013拼写检错任务中分享的训练数据
path13 = '../data/public_simple_data/rightSen13'
# sighan 2015拼写检错任务中分享的训练数据
path15 = '../data/public_simple_data/rightSen15'
# sighan 拼写检错任务中分享的测试数据
path_test = '../data/public_simple_data/rightTest'
texts13 = open(path13).read().split('\n')
texts15 = open(path15).read().split('\n')
texts_test = open(path_test).read().split('\n')
sentence = sentence + texts13 + texts15 + texts_test
for string in sentence:
    temp = list(string)
    str = ''
    for ch in temp:
        str = str+ch+' '
    # print(str)
    words.append(str)
model = Word2Vec(words, size=128, window=4, min_count=1, sg=1, workers=2)
model.save('wordVec_model/word2vecModel_pub')