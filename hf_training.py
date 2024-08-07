from datasets import load_dataset, load_from_disk, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from pprint import pprint
import evaluate
import numpy as np

START_FROM_ZERO = True
EPOCHS_TO_TRAIN = 2

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


dataset = dataset = load_from_disk('./news_dataset.hf')
# pprint(dataset)

model_name = 'neuralmind/bert-base-portuguese-cased'

pprint(list(labels_str2int.keys()))
classLabels = ClassLabel(
    num_classes=len(labels_str2int),
    names=list(labels_str2int.keys())
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(sample):
    # Preprocess the batch
    sample['label'] = [labels_str2int[categoria]
                       for categoria in sample['categoria']]

    return tokenizer(
        sample['texto'],
        padding='max_length',
        truncation=True,
        max_length=128
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)

if START_FROM_ZERO:
    model = AutoModelForSequenceClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased', num_labels=len(labels_str2int))
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        './news_model', num_labels=len(labels_str2int))
    

metric = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='training_output',
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=EPOCHS_TO_TRAIN,
    load_best_model_at_end=True,
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['valid'],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('news_model')
