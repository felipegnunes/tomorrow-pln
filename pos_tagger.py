import sys
import nltk
import pickle
import random
from pprint import pprint
from load_conllu import load_conllu
from nltk.tokenize import word_tokenize


class POSTagger:

    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def train(self, train_set):
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def token_features(self, sentence, token_index):
        token = sentence[token_index]['form']
        features = {}

        features['suffix1'] = token[-1:]
        features['suffix2'] = token[-2:]
        features['suffix3'] = token[-3:]

        if token_index == 0:
            features['previousToken'] = '<INICIO>'
        else:
            features['previousToken'] = sentence[token_index - 1]['form']

        return features

    def tag_sentence(self, sentence):
        tagged_sentence = []

        for i, token in enumerate(sentence):
            features = self.token_features(sentence, i)

            tagged_sentence.append(
                (token['form'], self.classifier.classify(features)))

        return tagged_sentence

    def tag_string(self, sentence_str):
        tokenized_sentence = word_tokenize(sentence_str, language='portuguese')
        tokenized_sentence = [{'form': token, 'feats': {}}
                              for token in tokenized_sentence]

        return self.tag_sentence(tokenized_sentence)

    def accuracy(self, test_tokens):
        return nltk.classify.accuracy(self.classifier, test_tokens)

    def save(self):
        f = open(f'{self.model_name}.pickle', 'wb')
        pickle.dump(self.classifier, f)
        f.close()

    def load(self):
        f = open(f'{self.model_name}.pickle', 'rb')
        self.classifier = pickle.load(f)
        f.close()


def find_context_tokens(sentences, pos_tagger):
    labeled_tokens = []
    for sentence in sentences:
        for i, token in enumerate(sentence):
            features = pos_tagger.token_features(sentence, i)

            labeled_tokens.append(
                (features, token.get('pos'))
            )

    return labeled_tokens


def main():
    args = sys.argv[1:]
    mode = None
    if not args:
       mode = 'train'
    elif args[0] == 'train' or args[0] == 'test':
        mode = args[0]
    else:
        exit(f'You started the script with invalid arguments {args}. Valid arguments: train or test.')

    print('Starting POS tagger script')

    pos_tagger = POSTagger('pos_tagger')
    if mode == 'test':
        pos_tagger.load()

    print('Loading CONLLU datasets')
    if mode == 'train':
        bosque = load_conllu('./conllu/bosque.conllu')
        labeled_tokens = find_context_tokens(bosque.sentences, pos_tagger)
    
    test = load_conllu('./conllu/test.conllu')
    test_tokens = find_context_tokens(test.sentences, pos_tagger)

    if mode == 'train':
        print('Start training of the POS tagger')
        pos_tagger.train(labeled_tokens)

    print(f'POS tagger accuracy in test set: {pos_tagger.accuracy(test_tokens)}')

    print('\n')

    random.shuffle(test.sentences)
    test_sentence = test.sentences[0]

    test_tags = [(token.form, token.pos) for token in test_sentence]
    print('Test sentence:')
    pprint(test_tags)

    print()

    print('Predicted:')
    test_predicted = pos_tagger.tag_sentence(test_sentence)
    pprint(test_predicted)

    hits = 0
    test_size = len(test_sentence)
    for i in range(test_size):
        if test_predicted[i][1] == test_tags[i][1]:
            hits += 1

    print(f'Accuracy on test sentence: {hits/test_size}')

    print()
    pprint(pos_tagger.tag_string('Meu nome é Felipe Guimarães. Eu sou um cara legal.'))

    if mode == 'train':
        pos_tagger.save()


if __name__ == '__main__':
    main()
