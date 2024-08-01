# Tomorrow -- Processamento de Linguagem Natural

Aqui estão as instruções para executar o código das atividades da disciplina:

## Atividade 01 -- Analisador Morfológico

Para utilizar o analisador morfológico, você pode executar:
```bash
python analisador_morfologico.py
```

Alternativamente, você pode importar a classe AnalisadorMorfologico para o seu próprio código:

```python
from analisador_morfologico import AnalisadorMorfologico
```

### Exemplo de uso
```python
model = AnalisadorMorfologico()

sentence = '''
        As métricas de classificação não se adequam à tarefa
        de modelagem de linguagem.
    '''

    print('Sentence=', sentence)
    sentence_tagged = model.tag(sentence)
    print('Sentence tagged=')
    pprint(sentence_tagged)
```
Resultado do exemplo acima

```python
[{'feats': {'Gender': 'Fem', 'Number': 'Plur'}, 'form': 'As', 'pos': 'DET'},
 {'feats': {'Gender': 'Fem', 'Number': 'Plur'},
  'form': 'métricas',
  'pos': 'NOUN'},
 {'feats': {}, 'form': 'de', 'pos': 'ADP'},
 {'feats': {'Gender': 'Fem', 'Number': 'Sing'},
  'form': 'classificação',
  'pos': 'NOUN'},
 {'feats': {}, 'form': 'não', 'pos': 'ADV'},
 {'feats': {'Gender': 'Masc', 'Number': 'Sing', 'PronType': 'Prs'},
  'form': 'se',
  'pos': 'PRON'},
 {'feats': {'Gender': 'Masc',
            'Person': '3',
            'Tense': 'Pres',
            'VerbForm': 'Fin'},
  'form': 'adequam',
  'pos': 'VERB'},
 {'feats': {}, 'form': 'à', 'pos': 'X'},
 {'feats': {}, 'form': 'tarefa', 'pos': 'PROPN'},
 {'feats': {}, 'form': 'de', 'pos': 'ADP'},
 {'feats': {'Gender': 'Fem', 'Number': 'Sing'},
  'form': 'modelagem',
  'pos': 'NOUN'},
 {'feats': {}, 'form': 'de', 'pos': 'ADP'},
 {'feats': {'Gender': 'Fem', 'Number': 'Sing'},
  'form': 'linguagem',
  'pos': 'NOUN'},
 {'feats': {}, 'form': '.', 'pos': 'PUNCT'}]
```

Se você quiser entender como funcionam os classificadores utilizados pelo AnalisadorMorfologico, veja features_classifier.py e pos_tagger.py.

## Atividade 02 -- Etiquetagem morfossintática

Para executar o código do etiquetador morfossintático você deve escolher entre duas opções: train ou test. Se não for provido nenhum argumento, ele
treina novamente o tagger.

```bash
python3 pos_tagger
python3 pos_tagger train
python3 pos_tagger test
```

Alternivamente, você pode importar o etiquetador para seu próprio código:

```python
from pos_tagger import POSTagger

# Utilizando o modelo já treinado, salvo em pos_tagger.pickle
pos_tagger = POSTagger('pos_tagger')
pos_tagger.load()

# A sentença pode ser uma lista de dicionários
test_predicted = pos_tagger.tag_sentence([{'form': 'Isso'}, {'form': 'é'}, {'form': 'legal'}, {'form': '.'}])

# Ou apenas uma string
sentence_tagged = pos_tagger.tag_string('Isso é uma sentença de teste.')
```