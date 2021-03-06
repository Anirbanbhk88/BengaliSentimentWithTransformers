# -*- coding: utf-8 -*-
"""XLM-RoBERTa_w_CNN_3class.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HebgoUXBWsZGpwkUn2KNlUsYqbmcMVna
"""

from google.colab import drive
drive.mount('/content/drive')

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

!pip install transformers

! pip install sentencepiece

from transformers import XLMRobertaTokenizer, XLMRobertaModel
MODEL_TYPE = 'xlm-roberta-large'
print('Loading XLMRoberta tokenizer...')
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
roberta = XLMRobertaModel.from_pretrained(MODEL_TYPE)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

max_input_length = 400

print(max_input_length)

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

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

base_path = '/content/drive/MyDrive/Independent Study-NLP/Bengali_Sentiment-master/' #'IAS/SEMESTER 4/IS-NLP/Bengali_Sentiment-master/'
model_name = 'XLM-RoBERTa-large_w_CNN_'
classification_class = '3class_'
result_path = 'Results/training_results/'
report_path = 'Results/classification_report/'
Dataset = 'ProthomAlo' #'BookReviews'#'YouTube'
from torchtext import datasets

train_data= data.TabularDataset(
    path= base_path +'Dataset/'+Dataset + '/train.csv',  format='CSV', skip_header=True,
    fields=[('text', TEXT), ('labels', LABEL)])


test_data= data.TabularDataset(
    path=base_path +'Dataset/'+ Dataset+ '/test.csv',  format='CSV', skip_header=True,
    fields=[('text', TEXT), ('labels', LABEL)])

# valid_data= data.TabularDataset(
#     path='/content/drive/My Drive/Research_Shanto/Datasets/SemEvel/proc_SemEvel_2016_devtest.csv',  format='CSV',skip_header=True,
#     fields=[('text', TEXT), ('labels', LABEL)])
train_data, valid_data = train_data.split(split_ratio=0.85,random_state = random.seed(SEED))

total_dataset_size = len(train_data) + len(test_data)

train_data.examples[0]

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

print(train_data.weights)

tokens = tokenizer.convert_ids_to_tokens(vars(valid_data.examples[6])['text'])
print(tokens)

print(vars(train_data.examples[6]))

tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])

print(tokens)

LABEL.build_vocab(train_data)

"""Mapping the Sentiment Labels to indices S.T -----------> 0: Negative, 1: Positive, 2: Neutral"""

print(LABEL.vocab.stoi)

print(LABEL)

BATCH_SIZE = 16

device = torch.device('cuda')

train_iterator,valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data, valid_data,test_data), 
    sort_key=lambda x: len(x.text),
    batch_size = BATCH_SIZE,
    device = device)

import torch.nn as nn
import torch.nn.init as init
class  CNN_Text(nn.Module):
    
    def __init__(self, roberta, output_dim, dropout, dropout_embed=0.1):
        super(CNN_Text, self).__init__()
        
        
        # V = args.embed_num
        self.roberta = roberta
        embedding_dim = roberta.config.to_dict()['hidden_size']

        D = 768
        C = output_dim
        Ci = 1
        Co = 200
        Ks = [3,3]
        init_weight_value = 2.0



        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                  padding=(K//2, 0), dilation=1, bias=False) for K in Ks])
        

        self.dropout = nn.Dropout(dropout)
        self.dropout_embed = nn.Dropout(dropout_embed)
        in_fea = len(Ks) * Co
        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def forward(self, x):
        # print(x)
        # print(x.size())
        # x = self.embed(x)  # (N,W,D)
        x = self.roberta(x)[0]
        # print(x.size())
        # x = self.dropout_embed(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        # print(x.device)
      
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        arr = []
        for i in x:
          shape = i.shape
          i = torch.reshape(i,(shape[0],shape[1],shape[2]*shape[3]))
          print(i.shape)
          arr.append(F.max_pool1d(i, i.size(2)).squeeze(2))

        x = arr#[F.max_pool1d(i, i.size(2)).squeeze(3) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc(x)
        return logit

OUTPUT_DIM = 4 #no of output_dimension = no of class+1
N_LAYERS = 1
DROPOUT = 0.5
model = CNN_Text(roberta, OUTPUT_DIM, DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

for name, param in model.named_parameters():                
    if name.startswith('roberta'):
        param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

for name, param in model.named_parameters():                
    if param.requires_grad:
        print(name)

import torch.optim as optim

# weight = 1./ torch.FloatTensor([ , ])
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

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

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
loss_train = []
loss_val = []
y_ = []
acc_train = []
acc_val = []
best_acc = float('inf')
best_valid_loss = float('inf')
epochs_ = []
learning_rate = 5e-4
wt_decay = 1e-6
optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=wt_decay)
MODEL_SAVE_PATH ='/content/drive/MyDrive/Independent Study-NLP/Bengali_Sentiment-master/Dataset/'+ model_name + classification_class+ Dataset+'.pt'
model_trained = False

if not model_trained:
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
    torch.save({
            'epoch': N_EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':loss_train,
            'val_loss': loss_val,
            'train_acc':acc_train,
            'val_acc':acc_val
            }, MODEL_SAVE_PATH)
    model_trained=True

"""Plot the Accuracy and loss

We'll load up the parameters that gave us the best validation loss and try these on the test set - which gives us our best results so far!
"""

if model_trained:
  checkpoint = torch.load(MODEL_SAVE_PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  N_EPOCHS = checkpoint['epoch']
  loss_train = checkpoint['train_loss']
  loss_val = checkpoint['val_loss']
  acc_train = checkpoint['train_acc']
  acc_val = checkpoint['val_acc']

# Create dataframes
train_val_loss_df = pd.DataFrame({'epochs':np.arange(N_EPOCHS),'loss_train':np.array(loss_train), 'loss_val':np.array(loss_val)})
train_val_acc_df = pd.DataFrame({'epochs':np.arange(N_EPOCHS),'acc_train':np.array(acc_train), 'acc_val':np.array(acc_val)})

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

test_loss, test_acc, all_predictions,targets = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

df_train_test_val_results = pd.DataFrame({'epochs':np.arange(N_EPOCHS).tolist(),'train_accuracy':acc_train, 'train_loss':loss_train, 'val_accuracy':acc_val, 'val_loss':loss_val, 'test_accuracy':test_acc, 'test_loss':test_loss,'num_classes':OUTPUT_DIM-1,'Train_Size':len(train_data)/total_dataset_size,'Test_Size':len(test_data)/total_dataset_size,'Valid_Size':len(valid_data)/total_dataset_size, 'Optimizer':'Adam', 'Learning_rate':learning_rate, 'Weight_decay':wt_decay, 'Loss_function':'CrossEntropyLoss','Batch_Size':BATCH_SIZE})
df_train_test_val_results

df_train_test_val_results.to_csv(base_path +result_path+ 'results_'+ model_name + classification_class + Dataset+'.csv')

"""### Test the model, make predications, Show classification Report"""

from sklearn.metrics import classification_report

categories = ['Negative', 'Positive', 'Neutral']

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

from collections import Counter
print(Counter(targets).keys()) # equals to list(set(words))
print(Counter(targets).values())

report = classification_report(all_predictions, targets, target_names=categories, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report

df_report.to_csv(base_path +report_path+'report_'+ model_name + classification_class + Dataset+'.csv')