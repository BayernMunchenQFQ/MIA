# -*- coding: utf-8 -*-
import math
from random import shuffle

from preprocessing import *
from models import create_model

import pandas as pd
import os
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np



def load_data(path, num=None):
    #print('load data:',path)
    data = pd.read_csv(path)
    #print("data:", data.shape)
    data_x = data['content'].tolist()
    data_y = data['label'].tolist()
    data_list = list(zip(data_x, data_y))
    shuffle(data_list)
    data_x[:], data_y[:] = zip(*data_list)
    #print(data_x[0])
    if num is not None:
        return data_x[:num], data_y[:num]
    else:
        return data_x, data_y

def load_data2(data_x, data_y, num=None):
    data_list = list(zip(data_x, data_y))
    shuffle(data_list)
    data_x[:], data_y[:] = zip(*data_list)
    if num is not None:
        return data_x[:num], data_y[:num]
    else:
        return data_x, data_y

def load_tokenizer(train_path='',val_path='',test_path='',max_length=200,nb_words=80000, start_char=1, oov_char=2, index_from=3, withraw=False):
    print('create token...')
    train_x,train_y=load_data(train_path)
    val_x,val_y=load_data(val_path)
    test_x,test_y=load_data(test_path)
    #print(train_x[:5])
    token = Tokenizer(lower=True)
    token.fit_on_texts(train_x + val_x)
    index_word = token.index_word
    print('voca size:',len(index_word))




    #sent=token.texts_to_sequences(sentences)
    train_x_token=token.texts_to_sequences(train_x)
    #train_x_token=train_x_token[:10000]
    val_x_token=token.texts_to_sequences(val_x)
    #val_x_token=val_x_token[:1000]
    test_x_token=token.texts_to_sequences(test_x)
    #test_x_token=test_x_token[:8000]
    # train_x_token+=val_x_token
    # train_y+=val_y

    if start_char == None:
        train_x_token = [[w + index_from for w in x] for x in train_x_token]
        val_x_token = [[w + index_from for w in x] for x in val_x_token]
        test_x_token = [[w + index_from for w in x] for x in test_x_token]
    else:
        train_x_token = [[start_char] +[w + index_from for w in x] for x in train_x_token]
        val_x_token = [[start_char] + [w + index_from for w in x] for x in val_x_token]
        test_x_token = [[start_char] +[w + index_from for w in x] for x in test_x_token]

    train_x_token = [[w if w < nb_words else oov_char for w in x] for x in train_x_token]
    val_x_token = [[w if w < nb_words else oov_char for w in x] for x in val_x_token]
    test_x_token = [[w if w < nb_words else oov_char for w in x] for x in test_x_token]

    train_x_token = pad_sequences(train_x_token, maxlen=max_length)
    val_x_token = pad_sequences(val_x_token, maxlen=max_length)
    test_x_token = pad_sequences(test_x_token, maxlen=max_length)
    # print(traindata.content)
    print('train size:',len(train_x_token),' val size:',len(val_x_token),' test size:',len(test_x_token))
    return train_x_token,train_y,val_x_token,val_y,test_x_token,test_y

def create_loader(inputs,label,batch_size=32,shuffle=True,num_worker=4):
    #print('data size:',len(inputs),' batch size:',batch_size)
    inputs=torch.LongTensor(inputs)
    lables=torch.LongTensor(label)
    data=Data.TensorDataset(inputs,lables)
    data_laoder=Data.DataLoader(dataset=data,batch_size=batch_size,shuffle=shuffle,num_workers=num_worker)
    return data_laoder

def create_attack_loader(inputs,label,batch_size=32,shuffle=True,num_worker=4):
    #print('data size:',len(inputs),' batch size:',batch_size)
    inputs=torch.FloatTensor(inputs)
    lables=torch.LongTensor(label)
    data=Data.TensorDataset(inputs,lables)
    data_laoder=Data.DataLoader(dataset=data,batch_size=batch_size,shuffle=shuffle,num_workers=num_worker)
    return data_laoder

def create_shadow_train_loader(train_x,train_y,model_num=1,batch_size=32):
    print('split shadow train data...')
    loaders=[]
    length=int(len(train_x)/model_num)
    for i in range(model_num):
        loaders.append(create_loader(train_x[length*i:length*i+length],train_y[length*i:length*i+length],batch_size=batch_size))
    return loaders


def save_model(model,path):
    torch.save(model.state_dict(),path)
    print('model saved')

def clipDataTopX(dataToClip, top=3):
    res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
    return res

def train_model(train_loader,val_loader,model_name='bilstm',model_path='',epoch=10,lr=0.0001,num_class=4,gpu_id=0):
    model=create_model(model_name=model_name,num_class=num_class)
    device = torch.device('cuda', gpu_id)
    if os.path.isfile(model_path):
        state = torch.load(model_path)
        try:
            model.load_state_dict(state)
        except:
            model = model.module
            model.load_state_dict(state)
        model=model.to(device)
        print('modeel file is used,not training...')
        return model
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bestacc = 0
    for k in range(epoch):
        print('Start epoch %d' %k)
        model.train()
        correct_train = .0
        train_loss = 0
        for dataid, data in enumerate(train_loader):
            inputs, target = data
            inputs, target = Variable(inputs), Variable(target)
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            #print(output.shape,target.shape)
            # break
            loss = F.cross_entropy(output, target)
            train_loss += loss.item()
            pred1 = output.data.max(1, keepdim=True)[1]
            correct_train += pred1.eq(target.data.view_as(pred1)).cpu().sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_train = correct_train / len(train_loader.dataset)
        avg_loss_train = train_loss / len(train_loader.dataset)
        print("Train", 'Epoch %d : Loss %.4f Accuracy %.5f' % (k, avg_loss_train, acc_train))

        correct = .0
        total_loss = 0
        model.eval()
        for dataid, data in enumerate(val_loader):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = correct / len(val_loader.dataset)
        avg_loss = total_loss / len(val_loader.dataset)
        print('Epoch %d : Loss %.4f Accuracy %.5f' % (k, avg_loss, acc))
        is_best = acc > bestacc
        print(acc, bestacc, is_best)
        if is_best:
            bestacc=acc
            save_model(model, model_path)
    state = torch.load(model_path)
    try:
        model.load_state_dict(state)
    except:
        model = model.module
        model.load_state_dict(state)
    model = model.to(device)
    print('modeel file is used, training finished...')
    return model

def train_mutl_model(train_loader,val_loader,model_num=1,model_name='bilstm',model_path='',epoch=10,lr=0.0001,num_class=4,gpu_id=0):
    models=[]
    device = torch.device('cuda', gpu_id)
    for i in range(model_num):
        print('train model ',i )
        model = create_model(model_name=model_name, num_class=num_class)
        if os.path.isfile(model_path+'shadow_model_'+str(i)+'.pkl'):
            state = torch.load(model_path+'shadow_model_'+str(i)+'.pkl')
            try:
                model.load_state_dict(state)
            except:
                model = model.module
                model.load_state_dict(state)
            model = model.to(device)
            print('modeel file is used,not training...')
            models.append(model)
            continue
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        bestacc = 0
        for k in range(epoch):
            print('Start epoch %d' % k)
            model.train()
            correct_train = .0
            train_loss = 0
            for dataid, data in enumerate(train_loader):
                inputs, target = data
                inputs, target = Variable(inputs), Variable(target)
                inputs, target = inputs.to(device), target.to(device)
                output = model(inputs)
                # print(output.shape,target.shape)
                # break
                loss = F.cross_entropy(output, target)
                train_loss += loss.item()
                pred1 = output.data.max(1, keepdim=True)[1]
                correct_train += pred1.eq(target.data.view_as(pred1)).cpu().sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc_train = correct_train / len(train_loader.dataset)
            avg_loss_train = train_loss / len(train_loader.dataset)
            print("Train", 'Epoch %d : Loss %.4f Accuracy %.5f' % (k, avg_loss_train, acc_train))

            correct = .0
            total_loss = 0
            model.eval()
            for dataid, data in enumerate(val_loader):
                inputs, target = data
                inputs, target = inputs.to(device), target.to(device)
                output = model(inputs)
                loss = F.cross_entropy(output, target)
                total_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            acc = correct / len(val_loader.dataset)
            avg_loss = total_loss / len(val_loader.dataset)
            print('Epoch %d : Loss %.4f Accuracy %.5f' % (k, avg_loss, acc))
            is_best = acc > bestacc
            print(acc, bestacc, is_best)
            if is_best:
                bestacc = acc
                save_model(model, model_path+'shadow_model_'+str(i)+'.pkl')
        state = torch.load(model_path+'shadow_model_'+str(i)+'.pkl')
        try:
            model.load_state_dict(state)
        except:
            model = model.module
            model.load_state_dict(state)
        model = model.to(device)
        print('modeel file is used, training finished...')
        models.append(model)
    return models



def get_attack_data(data_loader,model,label=0,number=1000,gpu_id=0):
    device = torch.device('cuda', gpu_id)
    x=None
    #y=None
    model.eval()
    for k,(inputs,labels) in enumerate(data_loader):
        inputs,labels=Variable(inputs).to(device),Variable(labels).to(device)
        output = model(inputs)

        output=output.cpu().detach().numpy()
        if k==0:
            x=output
            #y=np.ones(output.shape[0])

        else:
        #print(output.shape)
            x=np.concatenate((x,output))
            #y=np.concatenate((y,np.ones(output.shape[0])))
        #print(x.shape, y.shape)
        if x.shape[0]>=number:
            break
    x=x[:number]
    #y=y[:number]
    x=x.tolist()
    y=[label]*number
    return x,y


def cross_report(member_cross_list, none_member_cross_list, standard):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    accurracy  = 0
    for cross in member_cross_list:
        if cross < standard:
            tp += 1
        else:
            fp += 1
    for cross in none_member_cross_list:
        if cross >= standard:
            tn += 1
        else:
            fn += 1
    test_accurracy = (tp + tn) / (tp + tn + fp + fn)
    if test_accurracy > accurracy:
        accurracy = test_accurracy

    member_precious = tp / (tp + fn) if (tp + fn) != 0 else 111
    member_recall = tp / (tp + fp) if (tp + fp) != 0 else 111
    none_member_precious = tn / (tn + fp) if (tn + fp) != 0 else 111
    none_member_recall = tn / (tn + fn) if (tn + fn) != 0 else 111

    print("\t\tprecious\trecall")
    print("1\t\t{:.2f}\t\t{:.2f}".format(member_precious, member_recall))
    print("0\t\t{:.2f}\t\t{:.2f}".format(none_member_precious, none_member_recall))
    print("accurracy\t\t\t{:.2f}".format(accurracy * 100))


def entropy_accurracy(member_cross_list, none_member_cross_list, standard):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    accurracy  = 0
    for cross in member_cross_list:
        if cross < standard:
            tp += 1
        else:
            fp += 1
    for cross in none_member_cross_list:
        if cross >= standard:
            tn += 1
        else:
            fn += 1
    test_accurracy = (tp + tn) / (tp + tn + fp + fn)
    if test_accurracy > accurracy:
        accurracy = test_accurracy
    return accurracy

def cross_entropy_function(proba_list, reduction='none'):
    x_max = np.argmax(proba_list, axis=1)
    proba_list = torch.tensor(proba_list)
    x_max = torch.LongTensor(x_max)
    if reduction is None:
        cross_list = F.cross_entropy(proba_list, x_max)
    else:
        cross_list = F.cross_entropy(proba_list, x_max, reduction=reduction)
    cross_list = cross_list.tolist()
    return cross_list


def entropy_function(proba_list, reduction='none'):
    entropy_list = []
    for probas in proba_list:
        entropy = 0
        for p in probas:
            entropy += -1 * p * math.log(p, math.e)
        entropy_list.append(entropy)
    if reduction is None:
        return np.mean(entropy_list)
    return entropy_list


