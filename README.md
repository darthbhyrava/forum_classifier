## Health Forum Message Classification

### Project Description

Online health forum discussions are becoming first stop for patient or their relatives to seek advice on health related issues. Because of their free nature, administering these discussions require lots of manual effort. In this project, we utilize deep learning technique to classify online health discussion forum posts into one of the following 7 categories:

1. *Demographic (DEMO)*: Forums targeted towards specific demographic sub-groups characterized by age, gender, profession, ethnicity, etc.
2. *Disease (DISE)*: Forums related to a specific disease
3. *Treatment (TRMT)*: Forums related to a specific treatment or procedure
4. *Goal-oriented (GOAL)*: Forums related to achieving a health goal, such as weight management, exercise regimen, etc.
5. *Pregnancy (PREG)*: Forums related to pregnancy, including forums on difficulties with conception and concerns about mother and unborn child's heath during pregnancy
6. *Family support (FMLY)*: Forums related to issues of a caregiver (rather than a patient), such with support of an ill child or spouse.
7. *Socializing (SOCL)*: Forums related to socializing, including hobbies and recreational activities, rather than a specific health-related issue.


### Method
We performed a high-level processing of the given data. For each health forum message we had a category, title and question. We removed any special characters during cleaning. The average length of the titles was 5.25 and max length is 26 words. For the questions the average length was 157.5 and maximum length is 1568 words. 

Some manipulations performed on the input data are​:
1. Truncating
2. Padding
3. Taking Average length of all the forum messages and then fixing it as the size
for the input.
4. Use Attention

Now we have used 3 models​:

1. **SVM on Doc2Vec embeddings of the title+question.**
	Here we trained a doc2vec model on the training data title and questions and got 300 dimensional
	embeddings for each title and question for a health forum message. We used these 300 dimensional
	embeddings to train an SVM with the labels for each class from 1-7. This resulted in an accuracy of 38%.
2. **Bi-directional LSTM on word2vec embeddings of title words.**
	After the pre-processing step here we trained a  bi-directional LSTM model as shown in the figure below:
	The model consists of an Bidirectional LSTM layer (2 RNN layers) the output of which is fed into an
	Attention layer which is then forwarded to a Dense layer. Then finally we have a 7x1 output layer which
	gives us the respective probabilities of the forum message belonging to one of the 7  classes.
	The activation function for the dense layer was *relu*.​ The activation for the dense layer is *sigmoid*.
		- *Optimizer* - _AdaDelta_
		- *Loss* - _Categorical Cross Entropy._
		- *Training Accuracy* -  77%
		- *Validation Accuracy* -  47%
		- *Testing Accuracy* -  48.6%
3. **CNN model on embeddings of title and questions.**
	Finally we used a  CNN model for classification:
	- **Word embedding**
		Keras provides a convenient way to convert positive integer representations of words into a word
		embedding by an Embedding layer. We will map each word onto a 32 length real valued vector. We will constrain each message
		(title+question) to be 100 words, truncating long sentences and pad the shorter sentences with zero values.
	- **CNN**
		Convolutional neural networks excel at learning the spatial structure in input data add a  one-dimensional CNN and max pooling layers after the Embedding layer.
		Parameters:
		- filter size = 3
		- poo size=15
		- No. of filters =  256
		- activation function =  relu
		- loss function =  categorical_crossentropy
		- optimizer= adam
		- batch size =50
	- **LSTM**
		This learned spatial features may then be learned as sequences by an LSTM layer.
		- No. of memory units (neurons) = 100
		- Dense output layer
		- No. of neurons = 7
		- activation function = softmax
		- No.of epochs = 3
		
### Dataset 

We have used the ICHI 2016 dataset as described [here](http://www.ieee-ichi.org/healthcare_data_analytics_callenge.html)
It is a training dataset consisting of 8,000 questions (each with a post title and message text), and labeled with one of the seven categories described above, along with a test dataset of 3,000 questions that are unlabeled. 

### Results
CNN+LSTM: 56%
RNN: 49.8%
SVM: 38%


### Team Members
 - Siddharth Gairola ([@sidgairola19](https://github.com/sidgairo18))
 - Harshit Patni ([@patniharshit)](https://github.com/patniharshit))
 - Yojana Birje ([@yojana_birje](https://www.hackerrank.com/yojana_birje?hr_r=1))
 - Sriharsh Bhyravajjula ([@darthbhyrava](https://github.com/darthbhyrava))
