#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn.functional as F
import pandas as pd



SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[ ]:


#get_ipython().system('pip install transformers')


# In[ ]:


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[ ]:


init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)


# In[ ]:


max_input_length = 400

print(max_input_length)


# In[ ]:


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


# In[ ]:


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


# In[ ]:


base_path =  '/content/drive/MyDrive/Independent Study-NLP/Bengali_Sentiment-master/' #'home/9bhowmic/Bengali_Sentiment-master/'
result_path = 'Results/training_results/'
report_path = 'Results/classification_report/'
model_name = 'BERT_w_LSTM_'
classification_class = '2class_'
Dataset = 'YouTube'

from torchtext import datasets

train_data= data.TabularDataset(
    path= base_path +'Dataset/'+Dataset + '/youtube_drama_train.csv',  format='CSV', skip_header=True,
    fields=[('text', TEXT), ('labels', LABEL)])


test_data= data.TabularDataset(
    path=base_path +'Dataset/'+ Dataset+ '/youtube_drama_test.csv',  format='CSV', skip_header=True,
    fields=[('text', TEXT), ('labels', LABEL)])

# valid_data= data.TabularDataset(
#     path='/content/drive/My Drive/Research_Shanto/Datasets/SemEvel/proc_SemEvel_2016_devtest.csv',  format='CSV',skip_header=True,
#     fields=[('text', TEXT), ('labels', LABEL)])
train_data, valid_data = train_data.split(split_ratio=0.85,random_state = random.seed(SEED))


# In[ ]:


train_data.examples[0]


# In[ ]:


print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")


# In[ ]:


print(train_data.weights)


# In[ ]:



tokens = tokenizer.convert_ids_to_tokens(vars(valid_data.examples[6])['text'])
print(tokens)


# In[ ]:


print(vars(train_data.examples[6]))


# In[ ]:


tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])

print(tokens)


# In[ ]:


LABEL.build_vocab(train_data)


# Mapping the Sentiment Labels to indices S.T -----------> 0: Negative, 1: Positive

# In[ ]:


print(LABEL.vocab.stoi)


# In[ ]:


print(LABEL)


# In[ ]:


BATCH_SIZE = 32

device = torch.device('cuda')

train_iterator,valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data, valid_data,test_data), 
    sort_key=lambda x: len(x.text),
    batch_size = BATCH_SIZE,
    device = device)



# In[ ]:


from transformers import BertTokenizer, BertModel

bert = BertModel.from_pretrained('bert-base-multilingual-cased')


# In[ ]:


import torch.nn as nn
from torch.autograd import Variable
class BiLSTM(nn.Module):
    def __init__(self, bert, batch_size, hidden_dim, num_layers, output_dim, dropout, dropout_emb):
        super(BiLSTM, self).__init__()
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        C = output_dim
        self.dropout = nn.Dropout(dropout)
        self.dropout_embed = nn.Dropout(dropout_emb)
        self.bilstm = nn.LSTM(768, self.hidden_dim, num_layers=self.num_layers, bias=True, bidirectional=True,
                              dropout=dropout)
        
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
        self.hidden = self.init_hidden(self.num_layers, batch_size)
    

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)))

    def forward(self, x):
        x = self.bert(x)[0]
        # print("After Bert")
        # x = self.embed(x)
        # x = self.dropout_embed(x)
        # print("After dropout_embed")
        # x = x.view(len(x), x.size(1), -1)
        # x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, self.hidden = self.bilstm(x )
        # print("After Bilstm")
        # print(bilstm_out.size())
        # print(self.hidden)
        # print(bilstm_out.shape)

        # bilstm_out = torch.transpose(bilstm_out, 0, 1)
        # print("After Transpose 1")
        # print(bilstm_out.shape)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        # print("After Transpose 2")
        # print(bilstm_out.shape)
        bilstm_out = torch.tanh(bilstm_out)
        # print("After tanh")
        # print(bilstm_out.shape)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        # print("After maxpool")
        # print(bilstm_out.shape)
        bilstm_out = torch.tanh(bilstm_out)
        # print("After last tanh")
        
        # bilstm_out = self.dropout(bilstm_out)

        # bilstm_out = self.hidden2label1(bilstm_out)
        # logit = self.hidden2label2(F.tanh(bilstm_out))
        # print(bilstm_out.shape)

        logit = self.hidden2label(bilstm_out)

        return logit


# In[ ]:


HIDDEN_DIM = 100
OUTPUT_DIM = 3 #no of output_dimension = no of class+1
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.5

model = BiLSTM(bert, BATCH_SIZE, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM, DROPOUT, dropout_emb = 0.5)


# In[ ]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[ ]:


for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False


# In[ ]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[ ]:


for name, param in model.named_parameters():                
    if param.requires_grad:
        print(name)


# In[ ]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters(),lr=5e-4, weight_decay=1e-6)


# In[ ]:


# weight = 1./ torch.FloatTensor([ , ])
criterion = nn.CrossEntropyLoss()


# In[ ]:


model = model.to(device)
criterion = criterion.to(device)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:


N_EPOCHS = 10
loss_train = []
loss_val = []
y_ = []
acc_train = []
acc_val = []
best_acc = float('inf')
best_valid_loss = float('inf')
epochs_ = []

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
        #torch.save(model.state_dict(), '/content/drive/MyDrive/Independent Study-NLP/Bengali_Sentiment-master/Dataset/Gsuit_'+ model_name + classification_class+ Dataset+'.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    loss_val.append(valid_loss)
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    acc_val.append(valid_acc)
    epochs_.append(epoch+1)


# Plot the Accuracy and loss

# In[ ]:


# Create dataframes
train_val_loss_df = pd.DataFrame({'epochs':np.arange(N_EPOCHS),'loss_train':np.array(loss_train), 'loss_val':np.array(loss_val)})
train_val_acc_df = pd.DataFrame({'epochs':np.arange(N_EPOCHS),'acc_train':np.array(acc_train), 'acc_val':np.array(acc_val)})


# In[ ]:


# Plot the dataframes
plt.plot(train_val_acc_df['acc_train'])
plt.plot(train_val_acc_df['acc_val'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()

plt.plot(train_val_loss_df['loss_train'])
plt.plot(train_val_loss_df['loss_val'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()


# We'll load up the parameters that gave us the best validation loss and try these on the test set - which gives us our best results so far!

# In[ ]:


#model.load_state_dict(torch.load('/content/drive/MyDrive/Independent Study-NLP/Bengali_Sentiment-master/Dataset/Gsuit_'+ model_name + classification_class+ Dataset+'.pt'))


# In[ ]:


test_loss, test_acc, all_predictions,targets = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# In[ ]:


df_train_test_val_results = pd.DataFrame({'epochs':np.arange(N_EPOCHS).tolist(),'train_accuracy':acc_train, 'train_loss':loss_train, 'val_accuracy':acc_val, 'val_loss':loss_val, 'test_accuracy':test_acc, 'test_loss':test_loss})
df_train_test_val_results


# In[ ]:


df_train_test_val_results.to_csv(base_path+result_path+ model_name + classification_class + Dataset+'.csv')


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
print(Counter(targets).keys()) # equals to list(set(words))
print(Counter(targets).values())


# In[ ]:


report = classification_report(all_predictions, targets, target_names=categories, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report


# In[ ]:


df_report.to_csv(base_path+report_path+ model_name + classification_class + Dataset+'.csv')

