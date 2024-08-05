from datasets import load_dataset
from pprint import pprint
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
import evaluate
import numpy as np

# model_name = 'neuralmind/bert-base-portuguese-cased'
model_name = 'distilbert-base-uncased'

dataset = load_dataset('yelp_review_full')
# pprint(dataset)
# dataset['train'] = dataset['train'].select(range(2000))
# dataset['test'] = dataset['test'].select(range(2000))

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_dataset["train"].shuffle(
    seed=42).select(range(1000))
small_eval_dataset = tokenized_dataset["test"].shuffle(
    seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=5)


metric = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='test_trainer',
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=100,
    load_best_model_at_end=True,
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('my_model')
