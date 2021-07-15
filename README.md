# IMDB_Classifier

## Model
LSTM

## Hyper Parameters
lr = 1e-4

batch_size = 64

dropout_keep_prob = 0.5

embedding_size = 500

max_document_length = 500  # each sentence has until 500 words

max_size = 50000 # maximum vocabulary size

seed = 5

num_classes = 3

dev_size = 0.8

num_hidden_nodes = 100

hidden_dim2 = 128

num_layers = 4  # LSTM layers

bi_directional = False

num_epochs = 10

### Optimizer
Adam
### Criterion
CrossEntropyLoss
