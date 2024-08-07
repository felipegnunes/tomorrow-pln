import evaluate
import numpy as np

from datasets import load_dataset, DatasetDict, ClassLabel, load_from_disk
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification)
from pprint import pprint
from torch import nn

labels_str2int = {'tecnologia': 0,
                  'economia': 1,
                  'ciencia': 2,
                  'saude': 3,
                  'america_latina': 4,
                  'cultura': 5,
                  'brasil': 6,
                  'sociedade': 7,
                  'internacional': 8}

labels_int2str = {0: 'tecnologia',
                  1: 'economia',
                  2: 'ciencia',
                  3: 'saude',
                  4: 'america_latina',
                  5: 'cultura',
                  6: 'brasil',
                  7: 'sociedade',
                  8: 'internacional'}

dataset = load_from_disk('./news_dataset.hf')
print(dataset)

tokenizer = AutoTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased')

model = AutoModelForSequenceClassification.from_pretrained(
    './news_model/', num_labels=9)

test_dataset = dataset['test']

predictions = []
references = []
for i, test_data in enumerate(test_dataset):
    # if i == 10:
    #     break

    print('Progress {}/{}'.format(i + 1, len(test_dataset)))

    reference_int = labels_str2int[test_data['categoria']]
    references.append(reference_int)

    pt_batch = tokenizer(
        [test_data['texto']],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    outputs = model(**pt_batch)

    logits = nn.functional.softmax(outputs.logits, dim=-1)
    predicted_index = logits.max(1).indices.tolist()[0]
    predictions.append(predicted_index)

    # print(logits, labels_int2str[label_index])
    # print(f"titulo='{test_data['titulo']}' label={test_data['categoria']} predicted={labels_int2str[predicted_index]}")
    print(f"label={test_data['categoria']} predicted={labels_int2str[predicted_index]}")
    print()

# print(references)
# print()
# print(predictions)

# outputs = model(**pt_batch)
# predictions = nn.functional.softmax(outputs.logits, dim=-1)
# print(predictions)
# categories = [labels_int2str[index] for index in predictions.max(1).indices.tolist()]
# print([labels_int2str[index] for index in predictions.max(1).indices.tolist()])

metric = evaluate.load('accuracy')

final_score = metric.compute(predictions=predictions,
                             references=references)
print()
print(final_score)
