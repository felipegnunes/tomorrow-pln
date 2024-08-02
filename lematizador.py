from collections import defaultdict
import random
from load_conllu import load_conllu
from nltk.probability import FreqDist
from pprint import pprint


class Lematizador():

    def __init__(self) -> None:
        pass

    def lematizar(self, token):
        token = token['form'].lower()

        return token


def main():
    bosque = load_conllu('./conllu/bosque.conllu')
    test = load_conllu('./conllu/test.conllu')

    tokens = bosque.words
    random.shuffle(tokens)

    possible_swaps = defaultdict(lambda: set())

    for token in tokens:
        token_form = token['form'].lower()
        token_lemma = token['lemma'].lower()
        common_prefix = ''

        while len(token_form) > 0 and len(token_lemma) > 0 and token_form[0] == token_lemma[0]:
            common_prefix += token_lemma[0]
            token_lemma = token_lemma[1:]
            token_form = token_form[1:]

        print(
            token['form'],
            token['lemma'],
            common_prefix,
            '(',
            token['form'].removeprefix(common_prefix),
            '->',
            token['lemma'].removeprefix(common_prefix),
            ")"
        )

        to_remove = token['form'].lower().removeprefix(common_prefix)
        to_add = token['lemma'].lower().removeprefix(common_prefix)

        if to_remove != to_add:
            possible_swaps[token['pos']].add(
                (to_remove, to_add)
            )

    pprint(tokens[:5])
    pprint(possible_swaps)
    print(len(possible_swaps))

    exit()
    fdist = FreqDist()
    lemmaFreq = FreqDist()
    noLower = 0
    for word in bosque.words:
        fdist[word['pos']] += 1
        lemmaFreq[word['lemma']] += 1

        if not word['form'].islower():
            print(word['form'])
            noLower += 1

    print(f'Num. pos tags= {len(fdist.keys())}')
    pprint(fdist.most_common())
    print(f'Num. of lemmas={len(lemmaFreq.keys())}')
    pprint(lemmaFreq.most_common()[:100])

    print('Has uppercase: ', noLower)

    for word in bosque.words:
        word_form = word['form']  # .lower()
        word_lemma = word['lemma']

        if not word_lemma in word_form:
            print((word_lemma, word_form))

    print(len(bosque.words))


if __name__ == '__main__':
    main()
