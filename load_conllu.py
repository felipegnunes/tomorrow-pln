import conllu
import itertools as it

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CoNLLU:
   def __init__(self, files):
      self.words = []
      self.sentences = []
      for f in files:
         parsed = conllu.parse(open(f).read())
         sents = [[AttributeDict(form = token['form'], lemma=token['lemma'],pos=token['upos'],feats=token['feats']) for token in tokenlist if token['upos']!='_'] for tokenlist in parsed]
         self.sentences.extend(sents)
         self.words.extend([word for sent in sents for word in sent])
      self.pos_tags = set([word.pos for word in self.words])
      self.feats_dict ={pos:set(it.chain.from_iterable([list(word.feats.keys()) for word in self.words if word.pos==pos and word.feats!= None])) for pos in self.pos_tags}

def load_conllu(filename):
   return CoNLLU(files=[filename])