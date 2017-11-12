from gensim.models import Doc2Vec, doc2vec
from ast import literal_eval
import pdb
from tqdm import tqdm
import json
import pickle
from doc_helper import load_data_and_labels

train_file = "./dataset/ICHI2016-TrainData.tsv"
test_file = "./dataset/new_ICHI2016-TestData_label.tsv"

class Embedding():

    def __init__(self, sents, articleId):
        self.sents = sents
        self.articleId = articleId
        self.labelledSents = []

    def label(self):
        for i in range(len(self.sents) / 2):
            self.labelledSents.append(doc2vec.LabeledSentence(words=self.sents[2*i].split(), tags=['title_%s' % self.articleId[i]]))
            self.labelledSents.append(doc2vec.LabeledSentence(words=self.sents[2*i+1].split(), tags=['question_%s' % self.articleId[i]]))

    def train(self):
        self.model = Doc2Vec()
        self.model.build_vocab(self.labelledSents)
        for i in tqdm(range(10)):
            self.model.train(self.labelledSents)


if __name__ == '__main__':

    sents = []
    articleId = []

    s = load_data_and_labels()
    s2 = load_data_and_labels(test_file)

    for i, x in enumerate(s):
        print i, len(x[0]), len(x[1])
        sents.append(x[0])
        sents.append(x[1])
        articleId.append('train_'+str(i))

    for i, x in enumerate(s2):
        print i, len(x[0]), len(x[1])
        sents.append(x[0])
        sents.append(x[1])
        articleId.append('test_'+str(i))

    pdb.set_trace()

    article_embed = {}
    
    e = Embedding(sents, articleId)
    e.label()
    e.train()
    e.model.save('./data/embed_model')
    
    for k in tqdm(articleId):
        article_embed['title_%s' % k] = e.model.docvecs['title_%s' % k]
        article_embed['question_%s' % k] = e.model.docvecs['title_%s' % k]
   
    fp = open('./data/doc_embed.pkl', 'w')
    pickle.dump(article_embed, fp)

