import os
import torch
from torchtext.legacy import data
import torch.optim as optim
import gc
from google.colab import drive 
from transformers import BertTokenizer, BertModel

import tf_preprocess
import tf_Transformer
import tf_train_test
import dataset
import iterator

if __name__=='__main__':

    gc.collect()
    torch.cuda.empty_cache()
    drive.mount('/content/gdrive/')
    path = '/content/gdrive/My Drive/Colab Notebooks/IMDB_Classifier/data/'
    
    max_document_length = 500
    dev_size = 0.8
    batch_size = 32
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    TEXT = data.Field(use_vocab=False, tokenize = tf_preprocess.tokenize_and_cut, 
                  batch_first=True, 
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx
                  )
    LABEL = data.LabelField(dtype = torch.float)

    fields = [('text', TEXT), ('label', LABEL)]
    trainset,validset,testset = dataset.get_data(path,fields,dev_size,seed)

    TEXT.build_vocab(trainset, min_freq=5) # 단어 집합 생성
    LABEL.build_vocab(trainset)
    vocab_size = len(TEXT.vocab)

    train_iterator, valid_iterator, test_iterator = iterator.build_iterator(trainset,validset,testset,DEVICE,batch_size)

    bert = BertModel.from_pretrained('bert-base-uncased')
    model = tf_Transformer.Transformer(bert,
                    HIDDEN_DIM,
                    OUTPUT_DIM,
                    N_LAYERS,
                    BIDIRECTIONAL,
                    DROPOUT)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    
    tf_train_test.run_train(model,train_iterator,valid_iterator,optimizer,criterion)
    
    model.load_state_dict(torch.load('tut6-model.pt'))

    test_loss, test_acc = tf_train_test.evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')