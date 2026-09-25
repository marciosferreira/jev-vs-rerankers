# Experimento: JEV como portão na frente de um RAG top-1

Gerado em 2026-09-24 16:31. Base: `data.jsonl` (20 trechos). Perguntas: `eval_questions.json` (16). Juiz: `anthropic/claude-sonnet-5`. Gerador: `openai/gpt-4.1-mini`. Embeddings: `openai/text-embedding-3-small`. Uma execução.

**Hipótese:** um RAG top-1 costuma acertar; o JEV detecta os casos em que o trecho do top-1 não basta e os corrige, com custo e latência pequenos no caso comum.

- **top1:** o trecho mais parecido e resposta direta (baseline).
- **jev_gate:** o JEV julga cada vizinho numa única chamada (relevância + suficiência) e para quando os trechos mantidos bastam (suficiência >= 0.5); até 5 vizinhos; desiste após 3 trechos seguidos com relevância < 0.1.
- **sim_gate:** o mesmo portão usando similaridade de cosseno: top-1 se similaridade >= 0.719, senão os vizinhos com similaridade >= 0.609. Limiares ajustados nestas mesmas perguntas (vantagem para o controle).
- **topk_eq:** top-2, o mesmo número médio de trechos que o jev_gate examinou.
- **oracle:** os trechos gold como contexto (teto).

## 1. O portão sabe quando o top-1 não basta?

Verdade: o top-1 basta quando a pergunta tem resposta e seu único trecho gold está em 1º lugar. Casos em que o top-1 basta: 6 de 16.

| Portão | Acurácia | Deixou passar um top-1 ruim (falso positivo) | Barrou um top-1 bom (falso negativo) | Acertos (VP / VN) |
|---|---|---|---|---|
| jev_gate | 94% | 0 | 1 | 5 / 10 |
| sim_gate | 88% | 1 | 1 | 5 / 9 |

O falso positivo é o erro caro: o portão aprova um trecho ruim e o sistema responde errado ou incompleto. O falso negativo só custa latência (examina trechos a mais sem necessidade).

## 2. Qualidade e custo

| Métrica | top1 | jev_gate | sim_gate | topk_eq | oracle |
|---|---|---|---|---|---|
| Respostas corretas e completas | 56% | 94% | 75% | 94% | 94% |
| Nota do juiz (1 a 5) | 3.88 | 4.88 | 4.38 | 4.81 | 4.94 |
| Alucinação | 0% | 0% | 0% | 0% | 6% |
| Trechos examinados (média) | 1.00 | 2.19 | 3.50 | 2.00 | 1.25 |
| Trechos enviados ao LLM (média) | 1.00 | 1.25 | 1.06 | 2.00 | 1.25 |
| Trechos não-gold enviados (média) | 0.25 | 0.06 | 0.19 | 0.88 | 0.00 |
| Latência p50 (s) | 2.36 | 3.09 | 2.30 | 2.32 | 1.57 |
| Latência p95 (s) | 3.16 | 10.19 | 4.38 | 64.63 | 2.74 |
| Caminho rápido (1 trecho) | - | 31% | 38% | - | - |

## 3. Comparação pareada com o top1

Correta = o juiz considera todas as partes corretas e não há alucinação.

| Sistema | Corrigiu um erro do top1 | Estragou um acerto do top1 | McNemar p | Diferença de acurácia (IC 95%) |
|---|---|---|---|---|
| jev_gate | 6 | 0 | 0.031 | +38% (+12% a +62%) |
| sim_gate | 3 | 0 | 0.250 | +19% (+0% a +38%) |
| topk_eq | 6 | 0 | 0.031 | +38% (+12% a +62%) |
| oracle | 6 | 0 | 0.031 | +38% (+12% a +62%) |

### jev_gate contra os controles

Mostra se o ganho vem do JEV ou apenas de examinar mais trechos (topk_eq) ou de um portão barato (sim_gate).

| Controle | jev_gate acerta e controle erra | controle acerta e jev_gate erra | McNemar p | Diferença de acurácia (IC 95%) | Trechos enviados (jev / controle) | Latência p50 (jev / controle) |
|---|---|---|---|---|---|---|
| sim_gate | 3 | 0 | 0.250 | +19% (+0% a +38%) | 1.25 / 1.06 | 3.09 / 2.30 |
| topk_eq | 0 | 0 | 1.000 | +0% (+0% a +0%) | 1.25 / 2.00 | 3.09 / 2.32 |

## 4. Caso comum x caso raro

Separação pelo caminho que o jev_gate tomou.

| Caminho | n | Acurácia top1 | Acurácia jev_gate | Latência top1 (s) | Latência jev_gate (s) | Custo extra (s) |
|---|---|---|---|---|---|---|
| rápido (top-1 aprovado) | 5 | 100% | 100% | 2.43 | 2.93 | +0.50 |
| lento (top-1 barrado) | 11 | 36% | 91% | 2.41 | 4.06 | +1.65 |
| total | 16 | 56% | 94% | 2.42 | 3.71 | +1.29 |

## 5. Por pergunta

| # | Categoria | top-1 basta? | Portão JEV | Passos JEV (relevância/suficiência) | top1 | jev_gate | sim_gate | topk_eq | oracle |
|---|---|---|---|---|---|---|---|---|---|
| q01 | simples | sim | aprova | cristo-1 (0.99/0.94) | ✅ | ✅ | ✅ | ✅ | ✅ |
| q02 | simples | não | barra | brasilia-1 (0.67/0.14) brasilia-2 (0.97/0.83) | ❌ | ✅ | ✅ | ✅ | ✅ |
| q03 | simples | sim | aprova | pao-2 (0.99/0.95) | ✅ | ✅ | ✅ | ✅ | ✅ |
| q04 | simples | sim | aprova | cafe-2 (0.98/0.92) | ✅ | ✅ | ✅ | ✅ | ✅ |
| q05 | simples | sim | aprova | cristo-3 (0.98/0.92) | ✅ | ✅ | ✅ | ✅ | ✅ |
| q06 | simples | sim | aprova | amazonia-3 (0.98/0.86) | ✅ | ✅ | ✅ | ✅ | ✅ |
| q07 | composta | não | barra | cristo-1 (0.86/0.05) cristo-2 (0.97/0.89) | ❌ | ✅ | ❌ | ✅ | ❌ |
| q08 | composta | não | barra | pao-1 (0.87/0.04) pao-2 (0.97/0.96) | ❌ | ✅ | ✅ | ✅ | ✅ |
| q09 | multi-tema | não | barra | cristo-2 (0.94/0.02) cristo-1 (0.21/0.02) cristo-3 (0.13/0.02) cristo-5 (0.09/0.02) cafe-3 (0.97/0.04) | ❌ | ❌ | ❌ | ❌ | ✅ |
| q10 | multi-tema | não | barra | maracana-1 (0.92/0.03) brasilia-1 (0.98/0.96) | ❌ | ✅ | ❌ | ✅ | ✅ |
| q11 | multi-tema | não | barra | eiffel-2 (0.88/0.04) eiffel-1 (0.96/0.92) | ❌ | ✅ | ✅ | ✅ | ✅ |
| q12 | multi-tema | não | barra | cafe-1 (0.93/0.03) amazonia-1 (0.97/0.94) | ❌ | ✅ | ❌ | ✅ | ✅ |
| q13 | sem resposta | não | barra | amazonia-2 (0.02/0.01) amazonia-4 (0.03/0.01) amazonia-1 (0.02/0.01) | ✅ | ✅ | ✅ | ✅ | ✅ |
| q14 | sem resposta | não | barra | cristo-1 (0.03/0.01) cristo-2 (0.02/0.01) cristo-5 (0.02/0.01) | ✅ | ✅ | ✅ | ✅ | ✅ |
| q15 | sem resposta | não | barra | amazonia-4 (0.08/0.03) cristo-3 (0.02/0.02) brasilia-2 (0.02/0.02) | ✅ | ✅ | ✅ | ✅ | ✅ |
| q16 | parcial | sim | barra | amazonia-4 (0.86/0.04) maracana-1 (0.02/0.04) cristo-1 (0.01/0.03) brasilia-1 (0.02/0.02) | ✅ | ✅ | ✅ | ✅ | ✅ |

## 6. Respostas

### q01: Quando o Cristo Redentor foi inaugurado?

Referência: Em 12 de outubro de 1931.

**top1** (correta, nota 5, contexto: cristo-1, 2.51 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1].

Juiz: A resposta indica corretamente a data de inauguração conforme a referência e o contexto fornecido, sem inventar informações.

**jev_gate** (correta, nota 5, contexto: cristo-1, 2.67 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1].

Juiz: A resposta indica corretamente a data de inauguração, coincidindo com a referência e o contexto fornecido.

**sim_gate** (correta, nota 5, contexto: cristo-1, 2.3 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1].

Juiz: A resposta informa corretamente a data de inauguração, coincidindo com a referência e apoiada pelo contexto fornecido.

**topk_eq** (correta, nota 5, contexto: cristo-1, cristo-4, 2.85 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1].

Juiz: A resposta indica corretamente a data de inauguração, coincidindo com a referência e citando a fonte adequada.

**oracle** (correta, nota 5, contexto: cristo-1, 1.7 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1].

Juiz: A resposta indica corretamente a data de inauguração, coincidindo com a referência e o contexto fornecido.

### q02: Quem projetou os prédios de Brasília?

Referência: Oscar Niemeyer (Congresso Nacional, Palácio da Alvorada, Catedral Metropolitana).

**top1** (incorreta, nota 1, contexto: brasilia-1, 2.51 s)

> O contexto informa que o projeto urbanístico de Brasília foi feito por Lúcio Costa, mas não menciona quem projetou os prédios especificamente, então não sei essa parte. [brasilia-1]

Juiz: A resposta de referência indica que a base contém a informação de que Oscar Niemeyer projetou os prédios de Brasília, mas o sistema afirmou não saber essa informação, o que é considerado incorreto segundo as regras.

**jev_gate** (correta, nota 5, contexto: brasilia-1, brasilia-2, 3.08 s)

> Oscar Niemeyer projetou os principais edifícios de Brasília, como o Congresso Nacional, o Palácio da Alvorada e a Catedral Metropolitana [brasilia-2].

Juiz: A resposta corresponde exatamente à referência, citando Oscar Niemeyer e os três exemplos de edifícios, sem inventar informações adicionais.

**sim_gate** (correta, nota 5, contexto: brasilia-1, brasilia-2, 2.63 s)

> Oscar Niemeyer projetou os principais edifícios de Brasília, como o Congresso Nacional, o Palácio da Alvorada e a Catedral Metropolitana [brasilia-2].

Juiz: A resposta corresponde exatamente à referência, citando Oscar Niemeyer e os edifícios mencionados, sem inventar informações.

**topk_eq** (correta, nota 5, contexto: brasilia-1, brasilia-2, 2.76 s)

> Oscar Niemeyer projetou os principais edifícios de Brasília, como o Congresso Nacional, o Palácio da Alvorada e a Catedral Metropolitana [brasilia-2].

Juiz: A resposta corresponde exatamente à referência, citando Oscar Niemeyer e os edifícios mencionados, sem inventar informações.

**oracle** (correta, nota 5, contexto: brasilia-2, 1.6 s)

> Oscar Niemeyer projetou os principais edifícios de Brasília, como o Congresso Nacional, o Palácio da Alvorada e a Catedral Metropolitana [brasilia-2].

Juiz: A resposta corresponde exatamente à referência, citando Oscar Niemeyer e os mesmos exemplos de edifícios, sem inventar informações.

### q03: Qual a altura do morro do Pão de Açúcar?

Referência: 396 metros.

**top1** (correta, nota 5, contexto: pao-2, 2.05 s)

> A altura do morro do Pão de Açúcar é 396 metros [pao-2].

Juiz: A resposta informa corretamente os 396 metros, conforme a referência e o contexto fornecido, sem alucinações.

**jev_gate** (correta, nota 5, contexto: pao-2, 2.57 s)

> A altura do morro do Pão de Açúcar é 396 metros [pao-2].

Juiz: A resposta indica corretamente os 396 metros, conforme a referência e o contexto fornecido, sem inventar informações.

**sim_gate** (correta, nota 5, contexto: pao-2, 2.2 s)

> A altura do morro do Pão de Açúcar é 396 metros [pao-2].

Juiz: A resposta informa corretamente a altura de 396 metros, conforme a referência e o contexto fornecido.

**topk_eq** (correta, nota 5, contexto: pao-2, cristo-1, 2.02 s)

> A altura do morro do Pão de Açúcar é de 396 metros [pao-2].

Juiz: A resposta indica corretamente 396 metros, conforme a referência e o contexto fornecido.

**oracle** (correta, nota 5, contexto: pao-2, 1.61 s)

> A altura do morro do Pão de Açúcar é 396 metros [pao-2].

Juiz: A resposta informa corretamente a altura de 396 metros, conforme a referência e o contexto fornecido.

### q04: Qual estado brasileiro mais produz café?

Referência: Minas Gerais.

**top1** (correta, nota 5, contexto: cafe-2, 2.01 s)

> O estado brasileiro que mais produz café é Minas Gerais [cafe-2].

Juiz: A resposta corresponde exatamente à referência, indicando Minas Gerais e citando corretamente a fonte.

**jev_gate** (correta, nota 5, contexto: cafe-2, 3.24 s)

> O estado brasileiro que mais produz café é Minas Gerais [cafe-2].

Juiz: A resposta está correta, completa e alinhada com a referência e o contexto fornecido, citando corretamente Minas Gerais.

**sim_gate** (correta, nota 5, contexto: cafe-2, 2.37 s)

> O estado brasileiro que mais produz café é Minas Gerais [cafe-2].

Juiz: A resposta identifica corretamente Minas Gerais como o estado que mais produz café, alinhada com a referência e o contexto fornecido.

**topk_eq** (correta, nota 5, contexto: cafe-2, cafe-1, 3.26 s)

> O estado brasileiro que mais produz café é Minas Gerais [cafe-2].

Juiz: A resposta identifica corretamente Minas Gerais como o estado que mais produz café, alinhada com a referência e o contexto fornecido.

**oracle** (correta, nota 5, contexto: cafe-2, 1.09 s)

> Minas Gerais é o estado brasileiro que mais produz café [cafe-2].

Juiz: A resposta identifica corretamente Minas Gerais como o estado que mais produz café, conforme a referência e o contexto fornecido.

### q05: Quem esculpiu o rosto do Cristo Redentor?

Referência: O escultor romeno Gheorghe Leonida.

**top1** (correta, nota 5, contexto: cristo-3, 3.16 s)

> O rosto do Cristo Redentor foi esculpido pelo romeno Gheorghe Leonida [cristo-3].

Juiz: A resposta identifica corretamente Gheorghe Leonida como o escultor do rosto, coincidindo com a resposta de referência e o contexto fornecido.

**jev_gate** (correta, nota 5, contexto: cristo-3, 3.6 s)

> O rosto do Cristo Redentor foi esculpido pelo romeno Gheorghe Leonida [cristo-3].

Juiz: A resposta identifica corretamente Gheorghe Leonida como o escultor do rosto do Cristo Redentor, coincidindo com a referência e o contexto fornecido.

**sim_gate** (correta, nota 5, contexto: cristo-3, 3.48 s)

> O rosto do Cristo Redentor foi esculpido pelo romeno Gheorghe Leonida [cristo-3].

Juiz: A resposta identifica corretamente Gheorghe Leonida como o escultor do rosto do Cristo Redentor, coincidindo com a referência e o contexto.

**topk_eq** (correta, nota 5, contexto: cristo-3, cristo-5, 2.91 s)

> O rosto do Cristo Redentor foi esculpido pelo romeno Gheorghe Leonida [cristo-3].

Juiz: A resposta identifica corretamente o escultor Gheorghe Leonida, coincidindo com a referência e fundamentada no contexto fornecido.

**oracle** (correta, nota 5, contexto: cristo-3, 2.0 s)

> O rosto do Cristo Redentor foi esculpido pelo romeno Gheorghe Leonida [cristo-3].

Juiz: A resposta identifica corretamente Gheorghe Leonida como o escultor do rosto do Cristo Redentor, condizendo com a referência e o contexto fornecido.

### q06: Quais sistemas o INPE usa para monitorar o desmatamento da Amazônia?

Referência: PRODES e DETER, por satélite.

**top1** (correta, nota 5, contexto: amazonia-3, 2.43 s)

> O INPE usa os sistemas PRODES e DETER para monitorar o desmatamento na Amazônia brasileira [amazonia-3].

Juiz: A resposta identifica corretamente os sistemas PRODES e DETER, alinhando-se totalmente com a resposta de referência e o contexto fornecido.

**jev_gate** (correta, nota 5, contexto: amazonia-3, 2.56 s)

> O INPE usa os sistemas PRODES e DETER para monitorar o desmatamento da Amazônia brasileira [amazonia-3].

Juiz: A resposta identifica corretamente os sistemas PRODES e DETER, coincidindo com a referência e o contexto fornecido, sem inventar informações.

**sim_gate** (correta, nota 5, contexto: amazonia-3, 4.38 s)

> O INPE usa os sistemas PRODES e DETER para monitorar o desmatamento na Amazônia brasileira [amazonia-3].

Juiz: A resposta identifica corretamente os sistemas PRODES e DETER, com menção ao monitoramento por satélite, alinhando-se totalmente com a referência e o contexto fornecido.

**topk_eq** (correta, nota 5, contexto: amazonia-3, amazonia-1, 3.12 s)

> O INPE usa os sistemas PRODES e DETER para monitorar o desmatamento da Amazônia brasileira pelo satélite [amazonia-3].

Juiz: A resposta identifica corretamente os sistemas PRODES e DETER, condizendo totalmente com a referência e o contexto fornecido.

**oracle** (correta, nota 5, contexto: amazonia-3, 1.57 s)

> O INPE usa os sistemas PRODES e DETER para monitorar o desmatamento da Amazônia brasileira [amazonia-3].

Juiz: A resposta identifica corretamente PRODES e DETER como os sistemas usados pelo INPE, condizente com a referência e o contexto fornecido.

### q07: Quando o Cristo Redentor foi inaugurado e qual a sua altura?

Referência: Inaugurado em 12 de outubro de 1931; tem 30 metros de altura (sem o pedestal de 8 metros).

**top1** (incorreta, nota 3, contexto: cristo-1, 2.44 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1]. O contexto não informa a altura do monumento.

Juiz: A data de inauguração está correta, mas a resposta não informa a altura, que segundo a referência estava disponível na base, constituindo falha de recuperação.

**jev_gate** (correta, nota 5, contexto: cristo-1, cristo-2, 3.81 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931. A estátua tem 30 metros de altura, sem contar o pedestal de 8 metros [cristo-1][cristo-2].

Juiz: A resposta contém a data correta de inauguração e a altura correta com o detalhe do pedestal, coincidindo integralmente com a referência e o contexto.

**sim_gate** (incorreta, nota 3, contexto: cristo-1, 3.1 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1]. O contexto não informa qual é a sua altura.

Juiz: A resposta acerta a data de inauguração, mas erra ao dizer que não sabe a altura, já que a referência indica que essa informação está disponível na base.

**topk_eq** (correta, nota 5, contexto: cristo-1, cristo-2, 2.32 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931. A estátua tem 30 metros de altura, sem contar o pedestal de 8 metros [cristo-1] [cristo-2].

Juiz: A resposta informa corretamente a data de inauguração e a altura da estátua, ambas coincidindo com a referência e suportadas pelo contexto.

**oracle** (incorreta, nota 4, contexto: cristo-1, cristo-2, 1.57 s)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931. A estátua tem 30 metros de altura, sem contar o pedestal de 8 metros, totalizando 38 metros de altura incluindo o pedestal [cristo-1], [cristo-2].

Juiz: Data e altura da estátua estão corretas conforme a referência, mas a resposta adiciona um cálculo (38 metros com pedestal) que não está explicitamente no contexto, sendo uma extrapolação não solicitada.

### q08: Quando foi inaugurado o bondinho do Pão de Açúcar e qual a altura do morro?

Referência: O bondinho foi inaugurado em 1912; o morro tem 396 metros.

**top1** (incorreta, nota 3, contexto: pao-1, 2.36 s)

> O bondinho do Pão de Açúcar foi inaugurado em 1912. O contexto não informa a altura do morro [pao-1].

Juiz: A resposta acertou a data de inauguração (1912), mas errou ao dizer que não sabe a altura do morro, já que a referência indica que essa informação está na base (396 metros).

**jev_gate** (correta, nota 5, contexto: pao-1, pao-2, 4.37 s)

> O bondinho do Pão de Açúcar foi inaugurado em 1912 e o morro do Pão de Açúcar tem 396 metros de altura [pao-1][pao-2].

Juiz: A resposta informa corretamente ambos os dados (ano de inauguração 1912 e altura de 396 metros), condizentes com a referência e apoiados pelo contexto fornecido.

**sim_gate** (correta, nota 5, contexto: pao-1, pao-2, 2.14 s)

> O bondinho do Pão de Açúcar foi inaugurado em 1912 e o morro tem 396 metros de altura [pao-1][pao-2].

Juiz: A resposta contém corretamente a data de inauguração (1912) e a altura do morro (396 metros), ambas alinhadas com a referência e o contexto fornecido.

**topk_eq** (correta, nota 5, contexto: pao-1, pao-2, 2.23 s)

> O bondinho do Pão de Açúcar foi inaugurado em 1912 e o morro do Pão de Açúcar tem 396 metros de altura [pao-1] [pao-2].

Juiz: A resposta contém corretamente ambas as informações (ano de inauguração e altura do morro), condizentes com a referência e o contexto fornecido.

**oracle** (correta, nota 5, contexto: pao-1, pao-2, 2.74 s)

> O bondinho do Pão de Açúcar foi inaugurado em 1912 e o morro do Pão de Açúcar tem 396 metros de altura [pao-1] [pao-2].

Juiz: A resposta traz corretamente a data de inauguração (1912) e a altura do morro (396 metros), ambos consistentes com o contexto e a referência.

### q09: Qual a altura do Cristo Redentor, quem trouxe o café ao Brasil e qual a vazão do rio Amazonas?

Referência: O Cristo tem 30 metros (sem o pedestal de 8 m); o café foi trazido por Francisco de Melo Palheta, em 1727, da Guiana Francesa; o Amazonas despeja cerca de 209 mil metros cúbicos por segundo no Atlântico.

**top1** (incorreta, nota 2, contexto: cristo-2, 2.93 s)

> A altura do Cristo Redentor é de 30 metros, sem contar o pedestal de 8 metros, e seus braços se estendem por 28 metros [cristo-2]. Não sei quem trouxe o café ao Brasil e qual a vazão do rio Amazonas.

Juiz: A resposta acerta a altura do Cristo, mas erra ao dizer 'não sei' sobre café e Amazonas, pois a referência indica que essas informações estão disponíveis na base.

**jev_gate** (incorreta, nota 3, contexto: cristo-2, cafe-3, 5.47 s)

> A altura do Cristo Redentor é de 30 metros, sem contar o pedestal de 8 metros, e seus braços se estendem por 28 metros [cristo-2]. O café chegou ao Brasil em 1727, trazido da Guiana Francesa por Francisco de Melo Palheta [cafe-3]. Não sei a vazão do rio Amazonas, pois essa informação não está no contexto fornecido.

Juiz: As partes sobre o Cristo Redentor e o café estão corretas e completas, mas a resposta erroneamente diz não saber a vazão do Amazonas, quando a referência indica que essa informação está disponível na base (209 mil m³/s).

**sim_gate** (incorreta, nota 1, contexto: vazio, 0.82 s)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta de referência mostra que a base contém todas as informações solicitadas, mas o sistema não recuperou nenhuma delas, respondendo apenas que não encontrou trechos relevantes, o que é incorreto para todas as partes.

**topk_eq** (incorreta, nota 2, contexto: cristo-2, cristo-1, 2.55 s)

> A altura do Cristo Redentor é de 30 metros, sem contar o pedestal de 8 metros, totalizando 38 metros de altura incluindo o pedestal [cristo-2]. 
> 
> O contexto fornecido não informa quem trouxe o café ao Brasil, nem a vazão do rio Amazonas. Portanto, não sei responder essas partes da pergunta.

Juiz: A altura do Cristo foi respondida corretamente, mas o sistema disse 'não sei' para café e vazão do Amazonas, que a referência indica estarem disponíveis na base, configurando falha de recuperação.

**oracle** (correta, nota 5, contexto: cristo-2, amazonia-2, cafe-3, 1.82 s)

> A altura do Cristo Redentor é de 30 metros, sem contar o pedestal de 8 metros, e seus braços se estendem por 28 metros [cristo-2]. O café foi trazido ao Brasil por Francisco de Melo Palheta, vindo da Guiana Francesa, em 1727 [cafe-3]. A vazão do rio Amazonas é de cerca de 209 mil metros cúbicos por segundo [amazonia-2].

Juiz: As três partes foram respondidas corretamente e de forma completa, alinhadas com a resposta de referência e sem alucinações.

### q10: Em que ano o Maracanã foi inaugurado e em que ano Brasília foi inaugurada?

Referência: O Maracanã em 1950; Brasília em 1960 (21 de abril).

**top1** (incorreta, nota 2, contexto: maracana-1, 2.25 s)

> O Maracanã foi inaugurado em 1950 [maracana-1]. O contexto não informa o ano de inauguração de Brasília, portanto não sei essa parte.

Juiz: A resposta acerta o ano do Maracanã, mas afirma incorretamente que não sabe sobre Brasília, quando a referência indica que essa informação deveria estar disponível na base (falha de recuperação).

**jev_gate** (correta, nota 5, contexto: maracana-1, brasilia-1, 10.19 s)

> O Maracanã foi inaugurado em 1950 e Brasília foi inaugurada em 1960 [maracana-1] [brasilia-1].

Juiz: A resposta indica corretamente os anos de inauguração do Maracanã (1950) e de Brasília (1960), condizendo com a referência.

**sim_gate** (incorreta, nota 3, contexto: maracana-1, 2.51 s)

> O Maracanã foi inaugurado em 1950 [maracana-1]. O contexto não informa o ano de inauguração de Brasília.

Juiz: A parte sobre o Maracanã está correta, mas a resposta erra ao dizer que não sabe sobre Brasília, já que a referência indica que essa informação deveria estar disponível (falha de recuperação).

**topk_eq** (correta, nota 5, contexto: maracana-1, brasilia-1, 2.1 s)

> O Maracanã foi inaugurado em 1950 e Brasília foi inaugurada em 1960 [maracana-1] [brasilia-1].

Juiz: A resposta informa corretamente os anos de inauguração do Maracanã (1950) e de Brasília (1960), conforme a referência, sem inventar dados adicionais.

**oracle** (correta, nota 5, contexto: maracana-1, brasilia-1, 2.16 s)

> O Maracanã foi inaugurado em 1950 [maracana-1] e Brasília foi inaugurada em 1960 [brasilia-1].

Juiz: A resposta indica corretamente 1950 para o Maracanã e 1960 para Brasília, alinhando-se totalmente com a referência, embora não especifique o dia 21 de abril.

### q11: Que outra estrutura famosa Gustave Eiffel projetou e qual a altura da Torre Eiffel?

Referência: A estrutura interna da Estátua da Liberdade; a Torre Eiffel tem cerca de 330 metros.

**top1** (incorreta, nota 3, contexto: eiffel-2, 2.27 s)

> Gustave Eiffel projetou a estrutura interna da Estátua da Liberdade, em Nova York [eiffel-2]. O contexto não informa a altura da Torre Eiffel.

Juiz: A resposta identifica corretamente a Estátua da Liberdade, mas afirma incorretamente que não sabe a altura da Torre Eiffel, quando a referência indica que essa informação está na base (cerca de 330 metros).

**jev_gate** (correta, nota 5, contexto: eiffel-2, eiffel-1, 3.28 s)

> Gustave Eiffel projetou a estrutura interna da Estátua da Liberdade, em Nova York, e a Torre Eiffel tem cerca de 330 metros de altura [eiffel-2][eiffel-1].

Juiz: A resposta identifica corretamente a Estátua da Liberdade e a altura de 330 metros, ambas alinhadas com a referência e suportadas pelo contexto fornecido.

**sim_gate** (correta, nota 5, contexto: eiffel-2, eiffel-1, 2.24 s)

> Gustave Eiffel projetou a estrutura interna da Estátua da Liberdade, em Nova York. A Torre Eiffel tem cerca de 330 metros de altura [eiffel-2] [eiffel-1].

Juiz: A resposta menciona corretamente a Estátua da Liberdade e a altura de 330 metros, ambas presentes na referência e sem contradições ou invenções.

**topk_eq** (correta, nota 5, contexto: eiffel-2, eiffel-1, 2.31 s)

> Gustave Eiffel projetou a estrutura interna da Estátua da Liberdade, em Nova York, e a Torre Eiffel tem cerca de 330 metros de altura [eiffel-2][eiffel-1].

Juiz: A resposta identifica corretamente a Estátua da Liberdade e a altura de 330 metros, correspondendo integralmente à referência e ao contexto fornecido.

**oracle** (correta, nota 5, contexto: eiffel-1, eiffel-2, 1.52 s)

> Gustave Eiffel projetou a estrutura interna da Estátua da Liberdade, em Nova York, e a Torre Eiffel tem cerca de 330 metros de altura [eiffel-1][eiffel-2].

Juiz: A resposta identifica corretamente a Estátua da Liberdade como outra obra de Eiffel e informa a altura correta da Torre Eiffel, ambas alinhadas com o contexto e a referência.

### q12: Qual a participação do Brasil na produção mundial de café e em quantos países está distribuída a Floresta Amazônica?

Referência: Cerca de um terço da produção global de café; a Floresta Amazônica está distribuída entre nove países.

**top1** (incorreta, nota 3, contexto: cafe-1, 2.45 s)

> O Brasil é responsável por cerca de um terço da produção global de café [cafe-1]. Sobre a distribuição da Floresta Amazônica em países, o contexto não fornece essa informação.

Juiz: A resposta acertou a participação do Brasil na produção de café, mas falhou em informar que a Amazônia está distribuída em nove países, informação que a referência indica estar disponível na base.

**jev_gate** (correta, nota 5, contexto: cafe-1, amazonia-1, 3.09 s)

> O Brasil é responsável por cerca de um terço da produção mundial de café. A Floresta Amazônica está distribuída entre nove países [cafe-1][amazonia-1].

Juiz: A resposta contém corretamente ambas as informações: um terço da produção mundial de café e nove países da Amazônia, alinhadas com a referência e o contexto fornecido.

**sim_gate** (incorreta, nota 3, contexto: cafe-1, 2.67 s)

> O Brasil é responsável por cerca de um terço da produção global de café [cafe-1]. Não sei em quantos países está distribuída a Floresta Amazônica, pois essa informação não está presente no contexto fornecido.

Juiz: A parte sobre o café foi respondida corretamente, mas a parte sobre os nove países da Amazônia foi respondida com 'não sei', quando a referência indica que essa informação deveria estar disponível na base.

**topk_eq** (correta, nota 5, contexto: cafe-1, amazonia-1, 2.16 s)

> O Brasil é responsável por cerca de um terço da produção global de café e a Floresta Amazônica está distribuída entre nove países [cafe-1] [amazonia-1].

Juiz: A resposta cobre corretamente ambas as partes: participação de um terço na produção mundial de café e distribuição da Amazônia em nove países, conforme a referência e o contexto.

**oracle** (correta, nota 5, contexto: amazonia-1, cafe-1, 1.37 s)

> O Brasil é responsável por cerca de um terço da produção mundial de café [cafe-1]. A Floresta Amazônica está distribuída entre nove países [amazonia-1].

Juiz: A resposta cobre corretamente ambas as partes: participação de um terço na produção mundial de café e distribuição da Amazônia entre nove países, conforme a referência.

### q13: Qual a população de Manaus?

Referência: A base não contém essa informação. A resposta correta é dizer que não sabe.

**top1** (correta, nota 5, contexto: amazonia-2, 2.03 s)

> O contexto fornecido não informa a população de Manaus, portanto, não sei essa parte.

Juiz: A resposta corretamente afirma não saber a população de Manaus, condizendo com a referência de que essa informação não está na base.

**jev_gate** (correta, nota 5, contexto: vazio, 2.2 s)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta corretamente indica que não há informação disponível, alinhando-se com a referência de que a base não contém esse dado.

**sim_gate** (correta, nota 5, contexto: vazio, 0.78 s)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta corretamente indica que não encontrou a informação, alinhando-se com a referência que afirma que a base não contém esse dado.

**topk_eq** (correta, nota 5, contexto: amazonia-2, amazonia-4, 64.63 s)

> O contexto fornecido não informa a população de Manaus, portanto não sei essa parte. [amazonia-2] [amazonia-4]

Juiz: A base não contém a informação sobre a população de Manaus, e o sistema corretamente indicou que não sabe, sem inventar dados.

**oracle** (correta, nota 5, contexto: vazio, 0.0 s)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta corretamente indica que não encontrou informação, alinhando-se com a referência de que a base não contém esse dado.

### q14: Quanto custa o ingresso para visitar o Cristo Redentor?

Referência: A base não contém essa informação. A resposta correta é dizer que não sabe.

**top1** (correta, nota 5, contexto: cristo-1, 2.33 s)

> O contexto fornecido não informa o preço do ingresso para visitar o Cristo Redentor [cristo-1]. Portanto, não sei essa parte.

Juiz: A resposta corretamente indica que não sabe o preço do ingresso, alinhando-se com a resposta de referência que afirma que a base não contém essa informação.

**jev_gate** (correta, nota 5, contexto: vazio, 2.19 s)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta corretamente indica que não há informação disponível na base, alinhando-se à referência sem inventar dados.

**sim_gate** (correta, nota 5, contexto: cristo-1, 2.55 s)

> O contexto fornecido não informa o valor do ingresso para visitar o Cristo Redentor [cristo-1]. Portanto, não sei essa parte.

Juiz: A resposta corretamente indica que não sabe o valor do ingresso, alinhando-se à referência de que a base não contém essa informação, sem inventar dados.

**topk_eq** (correta, nota 5, contexto: cristo-1, cristo-2, 2.63 s)

> Não sei o custo do ingresso para visitar o Cristo Redentor com base no contexto fornecido.

Juiz: A resposta corretamente indica que não sabe o custo do ingresso, condizente com a referência de que a base não contém essa informação.

**oracle** (correta, nota 5, contexto: vazio, 0.0 s)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta corretamente indica que não há informação disponível, alinhando-se com a referência que afirma que a base não contém essa informação.

### q15: Quem foi o arquiteto do Teatro Amazonas?

Referência: A base não contém essa informação (ela só diz que o teatro foi inaugurado em 1896, no ciclo da borracha). A resposta correta é dizer que não sabe.

**top1** (correta, nota 5, contexto: amazonia-4, 2.02 s)

> O contexto fornecido não informa quem foi o arquiteto do Teatro Amazonas.

Juiz: A resposta corretamente afirma não saber quem foi o arquiteto, alinhando-se com a referência de que a base não contém essa informação.

**jev_gate** (correta, nota 5, contexto: vazio, 2.28 s)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta corretamente indica que não encontrou a informação, alinhando-se à referência que afirma que a base não contém o nome do arquiteto.

**sim_gate** (correta, nota 5, contexto: amazonia-4, 2.09 s)

> O contexto fornecido não informa quem foi o arquiteto do Teatro Amazonas [amazonia-4]. Portanto, não sei essa parte.

Juiz: A resposta corretamente afirma não saber quem foi o arquiteto, alinhando-se com a referência de que essa informação não está na base.

**topk_eq** (correta, nota 5, contexto: amazonia-4, cristo-3, 2.32 s)

> O contexto não informa quem foi o arquiteto do Teatro Amazonas, apenas menciona que foi inaugurado em 1896, durante o ciclo da borracha, com materiais importados da Europa [amazonia-4]. Portanto, não sei quem foi o arquiteto do Teatro Amazonas.

Juiz: A resposta corretamente indica que a base não informa o arquiteto do Teatro Amazonas, alinhando-se com a resposta de referência, sem inventar informações.

**oracle** (correta, nota 5, contexto: vazio, 0.0 s)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A referência confirma que a base não contém a informação sobre o arquiteto, e o sistema corretamente indicou não ter encontrado dados relevantes, sem inventar nada.

### q16: Quando o Teatro Amazonas foi inaugurado e quantos lugares ele tem?

Referência: Foi inaugurado em 1896. A base não informa a capacidade; a resposta correta diz que não sabe essa parte.

**top1** (correta, nota 5, contexto: amazonia-4, 2.95 s)

> O Teatro Amazonas foi inaugurado em 1896. O contexto não informa quantos lugares ele tem [amazonia-4].

Juiz: A resposta indica corretamente o ano de inauguração e, conforme a referência, não sabe a capacidade, pois essa informação realmente não está na base.

**jev_gate** (correta, nota 5, contexto: amazonia-4, 4.72 s)

> O Teatro Amazonas foi inaugurado em 1896 [amazonia-4]. Não há informação no contexto fornecido sobre a quantidade de lugares que ele tem.

Juiz: A resposta indica corretamente a data de inauguração (1896) e abstém-se corretamente sobre a capacidade, que não está na base conforme a referência.

**sim_gate** (correta, nota 5, contexto: amazonia-4, 2.3 s)

> O Teatro Amazonas foi inaugurado em 1896 durante o ciclo da borracha [amazonia-4]. O contexto não informa o número de lugares que ele tem.

Juiz: A resposta informa corretamente o ano de inauguração (1896) e admite corretamente que não sabe a capacidade, alinhando-se totalmente com a referência.

**topk_eq** (correta, nota 5, contexto: amazonia-4, maracana-1, 2.16 s)

> O Teatro Amazonas foi inaugurado em 1896 durante o ciclo da borracha [amazonia-4]. O contexto fornecido não informa quantos lugares ele tem.

Juiz: A resposta acerta a data de inauguração e corretamente afirma não saber a capacidade, que também não consta na referência.

**oracle** (correta, nota 5, contexto: amazonia-4, 1.44 s)

> O Teatro Amazonas foi inaugurado em 1896 [amazonia-4]. O contexto não informa quantos lugares ele tem.

Juiz: A resposta informa corretamente a data de inauguração e corretamente admite não saber a capacidade de lugares, alinhando-se totalmente com a referência.

## Limitações

- 16 perguntas numa base de 20 trechos escrita para o teste: serve para validar o experimento, não para provar a hipótese. Com esse n, o teste de McNemar dificilmente dá significância; ele está aqui para quando o experimento rodar numa base pública maior.
- A proporção de perguntas em que o top-1 não basta foi escolhida ao montar o conjunto, não reflete tráfego real.
- Os limiares do sim_gate e a parada antecipada do jev_gate foram escolhidos olhando estas mesmas perguntas. Numa base pública, devem ser ajustados num conjunto separado e congelados.
- O juiz é um LLM e pode errar.
- Com poucas perguntas, a latência p95 é praticamente o valor máximo e fica sensível a uma única chamada lenta ou repetida pela rede; a p50 é a medida confiável aqui.
