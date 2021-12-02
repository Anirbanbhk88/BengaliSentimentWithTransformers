#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torchtext

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[3]:


#get_ipython().system('pip install transformers')


# In[4]:


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[5]:


init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)


# In[6]:


max_input_length = 400

print(max_input_length)


# In[7]:


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


# In[8]:


#from torchtext.legacy import data
from torchtext.legacy import data

TEXT = data.Field(sequential= True,
                  batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField(sequential= False,dtype = torch.long)


# In[9]:


base_path =  '/home/9bhowmic/Bengali_Sentiment-master/' #'/home/anirban/IAS/SEMESTER 4/IS-NLP/Bengali_Sentiment-master/'
model_name = 'BERT_w_GRU_'
classification_class = '2class_'
result_path = 'Results/training_results/'
report_path = 'Results/classification_report/'
Dataset = 'ProthomAlo'

from torchtext import datasets

train_data= data.TabularDataset(
    path=base_path +'Dataset/Paper4_News_ProthomAlo/train_2class.csv',  format='CSV', skip_header=True,
    fields=[('text', TEXT), ('labels', LABEL)])


test_data= data.TabularDataset(
    path=base_path +'Dataset/Paper4_News_ProthomAlo/test_2class.csv',  format='CSV', skip_header=True,
    fields=[('text', TEXT), ('labels', LABEL)])

# valid_data= data.TabularDataset(
#     path='/content/drive/My Drive/Research_Shanto/Datasets/SemEvel/proc_SemEvel_2016_devtest.csv',  format='CSV',skip_header=True,
#     fields=[('text', TEXT), ('labels', LABEL)])
train_data, valid_data = train_data.split(split_ratio=0.85,random_state = random.seed(SEED))


# In[10]:


print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")


# In[11]:


print(train_data.weights)


# In[12]:



tokens = tokenizer.convert_ids_to_tokens(vars(valid_data.examples[6])['text'])
print(tokens)


# In[13]:


print(vars(train_data.examples[6]))


# In[14]:


tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])

print(tokens)


# In[15]:


LABEL.build_vocab(train_data)


# In[16]:


print(LABEL.vocab.stoi)


# In[17]:


print(LABEL)


# In[18]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# In[19]:


BATCH_SIZE = 32

device = torch.device('cuda')

train_iterator,valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data, valid_data,test_data), 
    sort_key=lambda x: len(x.text),
    batch_size = BATCH_SIZE, 
    device = device)



# In[20]:


from transformers import BertTokenizer, BertModel

bert = BertModel.from_pretrained('bert-base-multilingual-cased')


# In[21]:


import torch.nn as nn

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        

        


        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)


 
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)


        #output = [batch size, out dim]
        
        return output


# In[22]:


HIDDEN_DIM = 300
OUTPUT_DIM = 3 #output dimension = nof of classes + 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)


# In[23]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[24]:


for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False


# In[25]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[26]:


for name, param in model.named_parameters():                
    if param.requires_grad:
        print(name)


# In[27]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters(),lr=5e-4, weight_decay=1e-6)


# In[28]:


# weight = 1./ torch.FloatTensor([ , ])
criterion = nn.CrossEntropyLoss()


# In[29]:


model = model.to(device)
criterion = criterion.to(device)


# In[30]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.max(preds, 1)[1]
    # print("rounded_preds: ",rounded_preds)
    # print("y: ",y)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# In[31]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    # hid = model.initialize_hidden_state(BATCH_SIZE, device)
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        

        predictions = model(batch.text)
        
        loss = criterion(predictions, batch.labels)
        
        acc = binary_accuracy(predictions, batch.labels)
        
        loss.backward()

        optimizer.step()
        
        epoch_loss += loss.item()
        
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[32]:


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    targs = []
    # hid = model.initialize_hidden_state(BATCH_SIZE, device)
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            # hid= tuple([each.data for each in hid])

            predictions = model(batch.text)
            rounded_predictions = torch.max(predictions, 1)[1]
            # print("rounded_predictions: ",rounded_predictions)
            all_predictions.extend(rounded_predictions.cpu().detach().numpy())
            # print("all_predictions: ",all_predictions)

            targs.extend(batch.labels.cpu().detach().numpy())

            loss = criterion(predictions, batch.labels)
            
            acc = binary_accuracy(predictions, batch.labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), all_predictions, targs


# In[33]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[36]:


N_EPOCHS = 10
loss_train = []
loss_val = []
y_ = []
acc_train = []
acc_val = []
best_acc = float('inf')
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    y_.append(int(epoch+1))
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc ,a,b= evaluate(model, valid_iterator, criterion)
        
    end_time = time.time()
        
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_acc = valid_acc
        #torch.save(model.state_dict(), '/content/drive/MyDrive/Independent Study-NLP/Bengali_Sentiment-master/Dataset/Gsuit_BertGRU_2class.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    loss_val.append(valid_loss)
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    acc_val.append(valid_acc)


# We'll load up the parameters that gave us the best validation loss and try these on the test set - which gives us our best results so far!

# In[ ]:


#model.load_state_dict(torch.load('/content/drive/MyDrive/Independent Study-NLP/Bengali_Sentiment-master/Dataset/Gsuit_BertGRU_2class.pt'))


# In[ ]:


test_loss, test_acc, all_predictions,targets = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# In[ ]:


df_train_test_val_results = pd.DataFrame({'epochs':np.arange(N_EPOCHS).tolist(),'train_accuracy':acc_train, 'train_loss':loss_train, 'val_accuracy':acc_val, 'val_loss':loss_val, 'test_accuracy':test_acc, 'test_loss':test_loss})
df_train_test_val_results.to_csv(base_path +result_path+ model_name + classification_class + Dataset+'.csv')
# ### Test the model, make predications, Show classification Report

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


categories = ['Negative', 'Positive']


# In[ ]:


y_pred_list = []
y_test_target_list = []
with torch.no_grad():
  model.eval()
  for batch in test_iterator:
    y_test_pred = model(batch.text)
    rounded_predictions = torch.max(y_test_pred, 1)[1]
    # print("rounded_predictions: ",rounded_predictions)
    y_pred_list.extend(rounded_predictions.cpu().detach().numpy())
    y_test_target_list.extend(batch.labels.cpu().detach().numpy())

    


# In[ ]:


from collections import Counter
#print(Counter(targets).keys()) # equals to list(set(words))
#print(Counter(targets).values())


# In[ ]:


report = classification_report(all_predictions, targets, target_names=categories, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report


# In[ ]:


df_report.to_csv(base_path +report_path+ model_name + classification_class + Dataset+'.csv')


# In[ ]:




