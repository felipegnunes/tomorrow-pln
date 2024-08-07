from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

tokenizer = AutoTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased')

model = AutoModelForSequenceClassification.from_pretrained(
    './news_model/', num_labels=9)

# Retirado de https://www.bbc.com/portuguese
# Lá tem as mesmas categorias do dataset em que esse modelo foi treinado
texto = """
Hackers norte-coreanos estão tentando roubar segredos nucleares e militares de governos e empresas privadas em todo o 
mundo, alertam Reino Unido, Estados Unidos e Coreia do Sul.
Segundo esses governos, grupos conhecidos pelos nomes Andariel, Onyx Sleet e DarkSeoul, entre outros, estão atacando 
sistemas de defesa, aeroespaciais, nucleares e de engenharia para obter informações confidenciais.
Os hackers têm buscado informações nos mais diversos setores — desde o processamento de urânio até o desenvolvimento de 
tanques, submarinos e torpedos — e têm como alvo o Reino Unido, os Estados Unidos, a Coreia do Sul, o Japão, a Índia e 
outros países, segundo reportagem de Gordon Corera, correspondente de segurança da BBC.
Suspeita-se que eles já tenham atacado digitalmente bases aéreas dos EUA, a Nasa e empresas americanas ligadas à área de 
defesa.
O alerta parece ser um sinal de que a atividade destes grupos, que combinam espionagem e lucro, preocupa as autoridades 
devido ao seu impacto tanto na tecnologia sensível como na vida cotidiana. 
"""

pt_batch = tokenizer(
    [texto],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

outputs = model(**pt_batch)

logits = nn.functional.softmax(outputs.logits, dim=-1)
predicted_index = logits.max(1).indices.tolist()[0]

print('Predicted={}', labels_int2str[predicted_index])
