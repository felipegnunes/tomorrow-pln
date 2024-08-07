from datasets import load_dataset, DatasetDict

def prepare_dataset():
    dataset = load_dataset('celsowm/bbc_news_ptbr')

    train_devtest = dataset['train'].train_test_split(test_size=0.2)

    test_datasets = train_devtest['test'].train_test_split(test_size=0.5)

    dataset_splitted = DatasetDict({
        'train': train_devtest['train'],
        'valid': test_datasets['train'],
        'test': test_datasets['test']
    })

    return dataset_splitted


dataset = prepare_dataset()

dataset = dataset.shuffle(seed=55)
dataset = dataset.flatten_indices()  # rewrite the shuffled dataset on disk as contiguous chunks of data

print(dataset)

dataset.save_to_disk("news_dataset.hf")
