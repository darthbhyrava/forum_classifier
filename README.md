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
RNNs, Bi-directional RNNs, LSTMs, CNNs or an ensemble of these will be used to run experiments on the training data. The training data has a Forum Message along with it’s category/class. First we will remove stop-words, hyperlinks, and special characters from the text of the titles and the questions. Then we will convert each word into it’s corresponding word2vec vector which will give us a fixed length vector for each word in the question. Then we will concatenate the vectors for all the filtered words in the Forum Message and create a fixed size input along with appropriate padding. This concatenated vector is then fed in as input to a Neural Network and will then be trained with the given training data. This we will do for each Forum Message given in the training data. 
Other manipulations which can be performed on the input data is - 
Truncating
Padding
Taking Average length of all the forum messages and then fixing it as the size for the input.
Use Attention.
Finally we will use categorical-cross-entropy for classification.

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
