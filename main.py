import os
import torch
import torch.nn as nn
import gc
import dataset
from torchtext.legacy import data
import iterator
from model import LSTM
from train_test import run_train,evaluate
import sys

if __name__=='__main__':
    gc.collect()
    torch.cuda.empty_cache()
    path = sys.argv[1]
    # hyper-parameters:
    lr = 1e-4
    batch_size = 64
    dropout_keep_prob = 0.5
    embedding_size = 500
    max_document_length = 500  # each sentence has until 100 words
    max_size = 50000 # maximum vocabulary size
    seed = 5
    num_classes = 3
    dev_size = 0.8

    TEXT = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True,
                      fix_length=max_document_length)
    LABEL = data.Field(sequential=False, batch_first=True)

    fields = [('text', TEXT), ('label', LABEL)]
    trainset,validset,testset = dataset.get_data(path,fields,dev_size,seed)

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

    TEXT.build_vocab(trainset, min_freq=5)  # 단어 집합 생성
    LABEL.build_vocab(trainset)
    vocab_size = len(TEXT.vocab)

    train_iterator, valid_iterator, test_iterator = iterator.build_iterator(trainset,validset,testset,DEVICE,batch_size)

    num_hidden_nodes = 100
    hidden_dim2 = 128
    num_layers = 4  # LSTM layers
    bi_directional = False
    num_epochs = 10

    to_train = True
    pad_index = TEXT.vocab.stoi[TEXT.pad_token]
    gc.collect()
    torch.cuda.empty_cache()
    # Build the model
    lstm_model = LSTM(vocab_size, embedding_size, num_hidden_nodes, hidden_dim2, num_classes, num_layers,
                      bi_directional, dropout_keep_prob, pad_index).to(DEVICE)

    # optimization algorithm
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # train and evaluation
    if (to_train):
        # train and evaluation
        run_train(num_epochs, lstm_model, train_iterator, valid_iterator, optimizer, loss_func)

        # load weights
    lstm_model.load_state_dict(torch.load(os.path.join('./', "saved_weights_LSTM.pt")))
    # predict
    test_loss, test_acc = evaluate(lstm_model, test_iterator, loss_func)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
