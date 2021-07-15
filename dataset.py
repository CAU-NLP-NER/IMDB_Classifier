from torchtext.legacy import data

def get_data(path,fields,dev_size,seed):
    trainset, testset = data.TabularDataset.splits(
        path=path,
        train=path + 'train_data.csv',
        test=path + 'test_data.csv',
        format='csv',
        fields=fields,
        skip_header=False
    )
    trainset, validset = trainset.split(split_ratio=dev_size, random_state=random.seed(seed))
    return trainset,validset,testset