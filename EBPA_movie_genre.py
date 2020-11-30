import numpy as np
import pandas as pd
import string

pd.set_option('max_colwidth', 200)

%matplotlib inline 

#list of all labels
LABEL_MAP = ['Family', 'Animation', 'Documentary', 'Action', 'Crime', 'History', 'Fantasy', 'Romance', 'Horror', 'Science Fiction', 'War', 'Foreign', 'Adventure', 'Thriller', 'Western', 'Music', 'TV Movie', 'Mystery']

# print(len(LABEL_MAP))

#embeddings_dict contains GloVe word embeddings, the below code extracts those from a file
embeddings_dict = {}
with open("D:\\5th SEM\\NFT\\glove.6B.100d.txt", 'r', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

n_train = 2000         #no. of training examples
n_test = 120           #no. of test examples 
n_input = 100*174      #neural network input no. of nodes
n_hidden_1 = 128       #no. of nodes in hidden layer 1
n_hidden_2 = 64        #no. of nodes in hidden layer 2
n_output = 18          #neural network output no. of nodes
max_len = 174          #max length of a sentence to be taken into consideration
n_glove = 100          #dimension of glove embedding vector
sentences_train = np.zeros((n_train, n_input, 1))     #used to store all training sentence vectors
sentences_test = np.zeros((n_test, n_input, 1))       #used to store all testing sentence vectors
train_labels = np.zeros((n_train, n_output, 1))       #used to store all training labels(boolean vectors)
test_labels = np.zeros((n_test, n_output,1))          #used to store all testing labels(boolean vectors)

meta_train = pd.read_csv("movies_train.csv", usecols = ["overview", "genre"])
df_train = pd.DataFrame({"overview": meta_train["overview"], "genre": meta_train["genre"]})   #train dataframe

meta_test = pd.read_csv("movies_test.csv", usecols = ["overview", "genre"])
df_test = pd.DataFrame({"overview": meta_test["overview"], "genre": meta_test["genre"]})      #test dataframe 

def make_dataset(df, sentences, labels_bool):   #Makes train/test dataset(creates input vectors and labels from dataframe)
    size = sentences.shape[0]
    for i in range(size):
        sentence = df["overview"][i].split()
        labels = df["genre"][i]
        word_matrix = np.zeros((max_len, n_glove))
        j = 0
        while j < min(len(sentence), max_len):
            if sentence[j] in embeddings_dict.keys():
                word_matrix[j,:] = embeddings_dict[sentence[j]]
            j = j+1
        sentences[i,:,:] = word_matrix.reshape(n_input,1)
        
        labels_vector = np.zeros((n_output,1))
        for j,label in enumerate(LABEL_MAP):
            if label in labels:
                labels_bool[i,j,:] = 1
    return sentences, labels_bool

sentences_train, train_labels = make_dataset(df_train, sentences_train, train_labels)
sentences_test, test_labels = make_dataset(df_test, sentences_test, test_labels)

def sigmoid(a):     #activation function
    return (1.0/(1.0 + np.exp(-1.0 * a)))

def forward_prop(sentence, weights, Z1, Z2, Y):   #performs forward propagation for one input
    W1, W2, W3, b1, b2, b3 = weights
    Zin1 = np.dot(W1, sentence) + b1
    Z1 = sigmoid(Zin1)
    Zin2 = np.dot(W2, Z1) + b2
    Z2 = sigmoid(Zin2)
    Yin = np.dot(W3, Z2) + b3
    Y = sigmoid(Yin)
    return (Z1, Z2, Y)

def backprop(sentence, target, Z1, Z2, Y, weights, alpha):    #performs backpropagation for one input
    W1, W2, W3, b1, b2, b3 = weights
    del_Y = (target - Y) * Y * (1 - Y)
    del_Z2 = np.dot(W3.T, del_Y) * Z2 * (1 - Z2)
    del_Z1 = np.dot(W2.T, del_Z2) * Z1 * (1 - Z1)
    W3 = W3 + (alpha * np.dot(del_Y, Z2.T))
    b3 = b3 + (alpha * del_Y)
    W2 = W2 + (alpha * np.dot(del_Z2, Z1.T))
    b2 = b2 + (alpha * del_Z2)
    W1 = W1 + (alpha * np.dot(del_Z1, sentence.T))
    b1 = b1 + (alpha * del_Z1)
    weights = (W1, W2, W3, b1, b2, b3)
    return weights

alpha = 0.3                                   #learning rate
W_L1 = np.random.randn(n_hidden_1,n_input)        #weights between input and hidden layer
b_L1 = np.random.randn(n_hidden_1,1)          #bais terms for 1st hidden layer
W_L2 = np.random.randn(n_hidden_2,n_hidden_1)         #weights between 1st hidden layer and 2nd hidden layer
b_L2 = np.random.randn(n_hidden_2,1)          #bais terms for 2nd hidden layer
W_L3 = np.random.randn(n_output,n_hidden_2)          #weights between 2nd hidden layer and output
b_L3 = np.random.randn(n_output,1)           #bais terms for output layer
Z1 = np.zeros((n_hidden_1,1))                 #1st hidden layer vector
Z2 = np.zeros((n_hidden_2,1))                 #2nd hidden layer vector
Y = np.zeros((n_train,n_output,1))                #output vector
del_Y = np.zeros((n_output,1))               #error term for output layer
del_Z1 = np.zeros((n_hidden_1,1))             #error term for 1st hidden layer
del_Z2 = np.zeros((n_hidden_2,1))             #error term for 2nd hidden layer

weights = (W_L1, W_L2, W_L3, b_L1, b_L2, b_L3)    #tuple of weights
for j in range(30):   # epochs
    error = 0    
    for i in range(n_train):    #performs repeatedly forward and backpropagation for all inputs
        Z1, Z2, Y[i,:,:] = forward_prop(sentences_train[i,:,:], weights, Z1, Z2, Y[i,:,:])
        weights = backprop(sentences_train[i,:,:], train_labels[i,:,:], Z1, Z2, Y[i,:,:], weights, alpha)
        error = error + np.sum(0.5 * np.square(train_labels - Y))
    error = error/n_train
    print(error)

#now print accuracy
Y_pred = np.zeros((n_test,n_output,1))
for i in range(n_test):    #performs forward prop on test data
    _, _, Y_pred[i,:,:] = forward_prop(sentences_test[i,:,:], weights, Z1, Z2, Y_pred[i,:,:])
    
Y_pred = (Y_pred > 0.5)*1    #set predictions with probability > 0.5 as 1 
correct = np.sum(Y_pred == test_labels)    #count the correctly labelled classes
print(correct/(n_test*n_output))
