# IMDB_Classifier

module화, 필요없는 코드들 제거하였습니다.

오늘 내내 코랩에서 GPU 사용이 안돼서 최신 업데이트된 모델을 테스팅하지는 못했습니다.
learning_rate, dropout, batch_size, hidden_nodes는 accuracy가 가장 높게 나온 것을 선택했습니다.


지금까지 테스팅한 걸로는 Test accuracy가 기존의 baseline 코드 정도의 성능밖에 나오지 않는데, Colab GPU 다시 사용 가능하게 되면 파라미터 튜닝을 더 해보도록 하겠습니다.

## Model
LSTM

Changed Tokenizer(nltk.word_tokenize) when build_vocab()
## Hyper Parameters
lr = 1e-4

batch_size = 64

dropout_keep_prob = 0.5

embedding_size = 500 # max document length랑 길이 맞춤

max_document_length = 500  # 500개 넘는 텍스트도 있지만 거의 없기 때문에 500개로 잘라냄

seed = 5

num_classes = 1 # output_dim

dev_size = 0.8

num_hidden_nodes = 100

hidden_dim2 = 128 

num_layers = 4  # LSTM layers, 2개와 4개 테스팅을 했는데 성능에서 큰 차이가 없었습니다.

bi_directional = False

num_epochs = 10 # 7~9 정도의 epoch에서 보통 가장 높은 accuracy를 보였습니다. 그 이상으로 반복(learning_rate, dropout도 영향이 있었음)하면 overfitting이 심하게 일어남.

### Optimizer
Adam
### Criterion
CrossEntropyLoss
