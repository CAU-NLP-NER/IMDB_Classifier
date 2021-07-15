from torchtext.legacy import data

def build_iterator(trainset,validset,testset,DEVICE,batch_size):
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (trainset, validset, testset),
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),  # Sort the batches by text length size
        sort_within_batch=True,
        device=DEVICE)
    return train_iterator, valid_iterator, test_iterator
