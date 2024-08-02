from pprint import pprint
from nltk.tokenize import word_tokenize

from pos_tagger import POSTagger
from features_classifier import FeaturesClassifier, ALL_FEATURES, FEATURE_VALID_CATEGORIES
from lematizador import Lematizador

class AnalisadorMorfologico:

    def __init__(self):
        print('POSTagger loading')
        self.pos_tagger = POSTagger('pos_tagger')
        self.pos_tagger.load()

        self.feature_classifiers = {}

        for feature_name in ALL_FEATURES:
            print(f'Loading {feature_name} classifier')
            self.feature_classifiers[feature_name] = FeaturesClassifier(
                model_name=f'{feature_name}_classifier', feature_name=feature_name)
            self.feature_classifiers[feature_name].load()

        self.lematizador = Lematizador()

    def tag(self, sentence):
        sentence_tokens = word_tokenize(sentence, language='portuguese')
        sentence_tokens = [{'form': token, 'feats': {}}
                           for token in sentence_tokens]

        sentence_tags = self.pos_tagger.tag_sentence(sentence_tokens)

        for token, sentence_tag in zip(sentence_tokens, sentence_tags):
            token['pos'] = sentence_tag[1]

        # Tag each feature
        for feature_name in ALL_FEATURES:
            for token in sentence_tokens:
                if token['pos'] in FEATURE_VALID_CATEGORIES[feature_name]:
                    token['feats'][feature_name] = self.feature_classifiers[feature_name].classify(
                        token['form'])
                    
        for token in sentence_tokens:
            token['lemma'] = self.lematizador.lematizar(token)


        return sentence_tokens


def main():
    print('Iniciando analisador morfologico')

    model = AnalisadorMorfologico()

    sentence = '''
        As métricas de classificação não se adequam à tarefa
        de modelagem de linguagem.
    '''

    print('Sentence=', sentence)
    sentence_tagged = model.tag(sentence)
    print('Sentence tagged=')
    pprint(sentence_tagged)


if __name__ == '__main__':
    main()
