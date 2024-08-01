import sys
import pickle
import random
import nltk
from pprint import pprint
from load_conllu import load_conllu

ALL_FEATURES = [
    'Gender',
    'Number',
    'Person',
    'Tense',
    'VerbForm',
    'PronType',
]

FEATURE_VALID_CATEGORIES = {
    'Gender': set(['NOUN', 'ADJ', 'VERB', 'PRON', 'DET']),
    'Number': set(['NOUN', 'ADJ', 'PRON', 'DET']),
    'Person': set(['VERB']),
    'Tense': set(['VERB']),
    'VerbForm': set(['VERB']),
    'PronType': set(['PRON']),
}


class FeaturesClassifier:
    model_name = None
    classifier = None
    load = None

    def __init__(self, model_name, feature_name) -> None:
        self.model_name = model_name
        self.feature_name = feature_name

    def classify(self, token):
        features = self.token_features(token)
        return self.classifier.classify(features)

    def train(self, train_set):
        train_features = [(self.token_features(token), label)
                          for (token, label) in train_set]
        random.shuffle(train_features)

        self.classifier = nltk.NaiveBayesClassifier.train(train_features)

    def accuracy(self, test_set):
        test_features = [(self.token_features(token), label)
                         for (token, label) in test_set]

        return nltk.classify.accuracy(self.classifier, test_features)

    def token_features(self, token):
        features = {}

        features['suffix1'] = token[-1:]
        features['suffix2'] = token[-2:]
        features['suffix3'] = token[-3:]

        return features

    def save(self):
        f = open(f'{self.model_name}.pickle', 'wb')
        pickle.dump(self.classifier, f)
        f.close()

    def load(self):
        f = open(f'{self.model_name}.pickle', 'rb')
        self.classifier = pickle.load(f)
        f.close()


def find_relevant_tokens(tokens, feature_name):
    labeled_tokens = []
    for token in tokens:
        if token.get('feats') and token.get('feats').get(feature_name):
            labeled_tokens.append(
                (token.form, token.get('feats').get(feature_name)))

    return labeled_tokens


"""
Features to classify:
* Gender
* Number
* Person
* Tense
* VerbForm
* PronType
"""


def main():
    bosque = load_conllu('./conllu/bosque.conllu')
    test = load_conllu('./conllu/test.conllu')

    all_features = [
        'Gender',
        'Number',
        'Person',
        'Tense',
        'VerbForm',
        'PronType',
    ]

    feature_name = 'Gender'

    labeled_tokens = find_relevant_tokens(bosque.words, feature_name)
    test_tokens = find_relevant_tokens(test.words, feature_name)

    print(labeled_tokens[450:500])

    features_classifier = FeaturesClassifier(
        f'{feature_name}_classifier', feature_name)
    features_classifier.train(labeled_tokens)
    print(
        f'{feature_name} classifier accuracy: {features_classifier.accuracy(test_tokens)}')
    features_classifier.save()

    print()
    random.shuffle(test_tokens)
    pprint([{'form': token[0], 'predicted': features_classifier.classify(token[0]), 'expected': token[1]}
           for token in test_tokens[:25]])


def train_all_feature_models():
    bosque = load_conllu('./conllu/bosque.conllu')
    test = load_conllu('./conllu/test.conllu')

    for feature_name in ALL_FEATURES:
        print(f'Training {feature_name} classifier')

        labeled_tokens = find_relevant_tokens(bosque.words, feature_name)
        test_tokens = find_relevant_tokens(test.words, feature_name)

        features_classifier = FeaturesClassifier(
            f'{feature_name}_classifier', feature_name)
        features_classifier.train(labeled_tokens)
        print(
            f'{feature_name} classifier accuracy: {features_classifier.accuracy(test_tokens)}')

        features_classifier.save()

        print()


if __name__ == '__main__':
    args = sys.argv[1:]
    
    mode = None
    if not args:
       mode = 'train'
    elif args[0] == 'train' or args[0] == 'test':
        mode = args[0]
    else:
        exit(f'You started the script with invalid arguments {args}. Valid arguments: train or test.')
    
    if mode == 'train':
        train_all_feature_models()
    else:
        main()
    
