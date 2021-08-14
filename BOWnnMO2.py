import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataset

nltk.download('punkt')


f = open('E:\chat bot intern\week 4\cooking.stackexchange.txt','r')
file_content=f.readlines()

sentence = []
label = []
for r in file_content:
    f = r.strip().split()
    n = r.strip().split("__label__")
    m = n[-1].split()[0]
    n = n[-1][len(m)+1:]
    f = f[0][9:]
    sentence.append(n)
    label.append(f)

#################################################

test_data = sentence[12000:]
test_label = label[12000:]
sentence = sentence[0:12000]
label = label[0:12000]
test_data2 = []
test_label2 = []
for r in range(len(test_label)):
    if test_label[r] in label:
        test_label2.append(test_label[r])
        test_data2.append(test_data[r])

print(test_data2)
print(test_label2)        

###################################################




stemmer  = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(w):
    return stemmer.stem(w.lower())    

def bag_of_words(tokenized_sentence, words):
    
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag



all_words = []
tags = label
xy = []
for i in sentence:
    w = tokenize(i)
    all_words.extend(w)

for l in range(len(tags)):
    xy.append((tokenize(sentence[l]),tags[l]))

#print(all_words)
#print("---------------------------")
#print(tags)
#print("-----------------------------")
#print(xy)
ignore_words = ['?','!','.',',','a','of','the','on','in','{','}','/','#','$','%','&']
ignore_words2 = range(10000)

all_words = [stem(w) for w in all_words if w not in ignore_words and w not in ignore_words2]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(len(tags), "tags:", tags)
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(len(all_words), "unique stemmed words:", all_words)
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------")



X_train = []
Y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(Y_train)

print(len(X_train),X_train)
print(len(y_train),y_train)



class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]

    def __len__(self):
        return self.n_samples


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.sig(out)
        out = self.l2(out)
        out = self.sig(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


num_epochs = 1000
batch_size = 200
learning_rate = 0.01
input_size = len(X_train[0])
hidden_size = 100
output_size = len(tags)
print(input_size, output_size)


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')




print(f'final loss: {loss.item():.4f}')

model.eval()
accu = []

for d in range(len(test_data2)):
    # sentence = "do you use credit cards?"
    sentence = test_data2[d]

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    ac_tag = test_label2[d]
    
    if ac_tag == tag:
        accu.append(1)
    else:
        accu.append(0)

print((accu.count(1))/(len(accu)))