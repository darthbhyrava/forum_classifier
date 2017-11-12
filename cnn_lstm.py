# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
# from keras.datasets import imdb
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import re
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest

docs=[]
labels=[]
stop_words = ["a", "as", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "aint", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "arent", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "cmon", "cs", "came", "can", "cant", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldnt", "course", "currently", "definitely", "described", "despite", "did", "didnt", "different", "do", "does", "doesnt", "doing", "dont", "done", "down", "downwards", "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "far", "few", "ff", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had", "hadnt", "happens", "hardly", "has", "hasnt", "have", "havent", "having", "he", "hes", "hello", "help", "hence", "her", "here", "heres", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "id", "ill", "im", "ive", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isnt", "it", "itd", "itll", "its", "its", "itself", "just", "keep", "keeps", "kept", "know", "knows", "known", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldnt", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "ts", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "theres", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "theyd", "theyll", "theyre", "theyve", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "value", "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasnt", "way", "we", "wed", "well", "were", "weve", "welcome", "well", "went", "were", "werent", "what", "whats", "whatever", "when", "whence", "whenever", "where", "wheres", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whos", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "wont", "wonder", "would", "would", "wouldnt", "yes", "yet", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves", "zero", "coord", "gr", "com", "tr", "td", "nbsp", "http", "https", "www"]
stop_words_dict={}
for word in stop_words:
	stop_words_dict[word]=1

f = open("./dataset/ICHI2016-TrainData.tsv","r")
cnt=0;
max1=0
for line in f:
	if cnt==0:
		cnt+=1
		continue
	l=line.strip().split('\t')
	str1=re.sub('[^a-zA-Z0-9\t\n\.]', ' ',l[1])
	str2=re.sub('[^a-zA-Z0-9\t\n\.]', ' ',l[2])
	str11=" "
	for l1 in str1.split(' '):
		if len(l1)>0 and (stop_words_dict.has_key(l1.lower().strip()))==False:
			str11=str11+' '+l1.lower().strip()
	for l2 in str2.split(' '):
		if len(l2)>0 and (stop_words_dict.has_key(l2.lower().strip()))==False:
			str11=str11+' '+l2.lower().strip()
	docs.append(str11)
	labels.append(l[0].strip())

encoder = LabelEncoder() # convert strings to int
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
y_train = np_utils.to_categorical(encoded_Y)

vocab_size = 50000
encoded_docs = [one_hot(d, vocab_size) for d in docs]
max_length =100
X_train = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


test_docs=[]
test_labels=[]
f = open("./dataset/new_ICHI2016-TestData_label.tsv","r")
cnt=0;
for line in f:
	if cnt==0:
		cnt+=1
		continue
	l=line.strip().split('\t')
	str1=re.sub('[^a-zA-Z0-9\.\n\t]', ' ',l[1])
	str2=re.sub('[^a-zA-Z0-9\.\n\t]', ' ',l[2])
	str11=" "
	for l1 in str1.split(' '):
		if len(l1)>0 and (stop_words_dict.has_key(l1.lower()))==False:
			str11=str11+' '+l1.lower().strip()
	for l2 in str2.split(' '):
		if len(l2)>0 and (stop_words_dict.has_key(l2.lower()))==False:
			str11=str11+' '+l2.lower().strip()
	test_docs.append(str11)
	test_labels.append(l[0].strip())

test_encoder = LabelEncoder() # convert strings to int
test_encoder.fit(test_labels)
test_encoded_Y = test_encoder.transform(test_labels)
# convert integers to dummy variables (i.e. one hot encoded)
y_test = np_utils.to_categorical(test_encoded_Y)

test_encoded_docs = [one_hot(d, vocab_size) for d in test_docs]
X_test = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')



top_words = 50000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# print X_train
# truncate and pad input sequences
# max_review_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model

#'''
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_length))
model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=15))
model.add(LSTM(32))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=50)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100)) #'''

