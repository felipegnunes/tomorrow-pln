# Tomorrow -- Processamento de Linguagem Natural

Aqui estão as instruções para executar o código das atividades da disciplina:

## Atividade 01 -- Analisador Morfológico

Para utilizar o analisador morfológico, você pode executar:
```bash
python3 analisador_morfologico.py
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

## Atividade 03 -- Classificação utilizando transformers

Obs.: Professor, não precisa baixar o dataset nem treinar o modelo, já estou enviando-os na pasta.

Para não haver mistura entre treino, validação e teste, vamos primeiro baixar o dataset do Hugging Face e guardá-lo pré-processado:
```bash
python3 hf_process_dataset.py
```

O dataset pré-processado fica salvo em news_dataset.hf.

Para medir a acurácia do modelo salvo localmente em news_model/:
```bash
python3 hf_testing.py
```

Se você quiser treinar o modelo do zero. Altere as seguintes variáveis globais em hf_training.py para:
```python
    START_FROM_ZERO = True
    EPOCHS_TO_TRAIN = 1
```

Caso queira continuar o treinamento de onde ele parou, utilize:
```python
    START_FROM_ZERO = False
    EPOCHS_TO_TRAIN = 1
```

Então execute:
```bash
python3 hf_training.py
```

Se você quiser utilizar um texto da sua preferência, veja um exemplo de uso em **hf_classification.py**.

O modelo base utilizado foi o BERTimbau Base (neuralmind/bert-base-portuguese-cased). Esse modelo foi treinado no dataset **celsowm/bbc_news_ptbr** do Hugging Face. 


Você pode encontrar notícias para testar em https://www.bbc.com/portuguese, elas são
da mesma fonte de dados do dataset, portanto separadas pelas mesmas categorias.

