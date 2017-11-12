import json
import pdb
import numpy as np
import pickle as pkl
from ast import literal_eval
import json
from tqdm import tqdm
import keras
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN, Dropout
from keras.layers.merge import concatenate, dot, multiply, add
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from attention import AttentionWithContext
from data_helper import load_data_and_labels

print ("Works")
train_file = "./dataset/ICHI2016-TrainData.tsv"
test_file = "./dataset/new_ICHI2016-TestData_label.tsv"

class RNN_Model:

    def __init__(self, normal_file, weird_file, title_max, word_embed_size):

        self.normal_file = normal_file
        self.weird_file = weird_file
        self.title_max = title_max
        self.word_embed_size = word_embed_size
        self.train = []
        self.train_truth = []
        self.test = []
        self.test_truth = []
        self.word_embed = json.load(open('glove_embed.json'))


    def data_handler(self):
        
        normal_data = []

        x, y = load_data_and_labels(test_file)
        
        data_embed = []
        for title_d in tqdm(x):
                
            temp = []
            title = title_d.strip('\'').split()
   
            for word in title[:15]:
                try:
                    temp.append(self.word_embed[word.lower()])
                except:
                    temp.append([0]*self.word_embed_size)
            for i in range(self.title_max - len(title)):
                temp.append([0]*self.word_embed_size)
            data_embed.append(temp)

	#self.truth.append(class_id)  add here class id

	#splitting data into training and validation here. 
	#change below code for training and validation by adding classes. He'd done 0 for normal and 1 for weird.
        size = len(data_embed)
        self.train += data_embed[:int(size*1.0)]
        self.test += data_embed[:int(size*1.0)]

        self.train_truth += y[:int(size*1.0)]
        self.test_truth += y[:int(size*1.0)]
        
        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.train_truth = np.array(self.train_truth)
        self.test_truth = np.array(self.test_truth)

        print("Data handler done")

    def create_model(self):

        title_words = Input(shape=(self.title_max, self.word_embed_size))
        
        #lstm_layer = LSTM(64, return_sequences=False)(title_words)
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(title_words)
        attention_layer = AttentionWithContext()(lstm_layer)
        dropout = Dropout(0.2)(attention_layer)
        dense = Dense(32, activation='relu')(dropout)
        #output = Dense(1, activation='sigmoid')(dense)
        output = Dense(7, activation='sigmoid')(dense)

        self.model = Model(inputs=[title_words], outputs=output)
        #self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        

    def fit_model(self, inputs, outputs, epochs):
        filepath="./weights/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        print("Yep")
        self.model.fit(inputs, outputs, validation_split=0.2, epochs=epochs, callbacks=callbacks_list, verbose=1)

    
def train(normal_file, weird_file, title_max, embed_size):
    model = RNN_Model(normal_file, weird_file, title_max, embed_size)
    print("1")
    model.data_handler()
    model.create_model()
    print("2")
    model.model.summary()
    print("3")
    model.fit_model([model.train], model.train_truth, 30)
    print("4")


def test(normal_file, weird_file, title_max, embed_size):
    model = RNN_Model(normal_file, weird_file, title_max, embed_size)
    model.data_handler()
    model.create_model()
    model.model.summary()
    model.model.load_weights('./weights/weights-05-1.35.hdf5')
    out = model.model.predict([model.test])

    hit = 0 
    for i in range(out.shape[0]):

        print (out[i], model.test_truth[i])
        temp_out = list(out[i])
        temp_truth = list(model.test_truth[i])
        if temp_out.index(max(temp_out)) == temp_truth.index(max(temp_truth)):
            hit += 1

    print (hit,out.shape[0], "Testing Accuracy", float(hit)/out.shape[0]*100)


if  __name__ == '__main__':

    #train('train_file', 'test_file', 15, 300)
    test('train_file', 'test_file', 15, 300)

