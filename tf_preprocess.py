from transformers import BertTokenizer

def tokenize_and_cut(sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens
