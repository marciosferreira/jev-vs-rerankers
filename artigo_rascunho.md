# JEV contra rerankers: ~8× menos alucinações em RAG a um sétimo do custo

## Um modelo de decisão, e não um reranker, deve decidir o que o seu RAG envia ao LLM

> **Rascunho** (versão em português; a versão para o Medium está em `artigo_medium_en.md`). Números de uma execução no SQuAD 2.0 (300 perguntas de teste).

**Autor:** Marcio Ferreira
**Data:** setembro de 2026

> **Antes de começar.** Este artigo não pretende parecer um texto escrito por mim. O objetivo é divulgar um experimento que testou uma ideia minha e cujos resultados, dentro dos limites descritos adiante, a sustentam. Para publicar os resultados mais rápido, o texto foi rascunhado por IA (Claude Opus 5.5) e revisado e editado por mim. O código dos experimentos também foi escrito com ela, sob a minha direção. A ideia, a tese e as decisões são minhas, e todos os números vêm do código e dos resultados publicados no repositório. Mais detalhes em "Como este trabalho foi feito", no fim.
>
> **Conflito de interesse:** não tenho vínculo com a TypeSafe, a Cohere, a Anthropic, a OpenAI ou o OpenRouter, nem recebi financiamento ou créditos deles. Todo o uso de API foi pago por mim (a Cohere, pela chave de teste gratuita).

---

## Resumo

Em sistemas de RAG (*Retrieval-Augmented Generation*), o recuperador por embeddings traz trechos **semanticamente parecidos** com a pergunta, e não necessariamente trechos que **respondem** a ela. Quando a resposta não está na base, o LLM recebe um contexto plausível mas inútil, e é aí que surge boa parte das alucinações. Os rerankers, a solução usual, melhoram a **ordem** dos trechos, mas são usados para devolver sempre os *k* primeiros e não sabem dizer "nenhum destes serve".

Avaliamos o **JEV** (TypeSafe), um modelo de decisão que responde perguntas de sim ou não com uma probabilidade, como filtro do contexto enviado ao LLM. O teste teve duas etapas, ambas com 300 perguntas do SQuAD 2.0: uma de **recuperação** (quem separa melhor "responde" de "só parece") e uma **ponta a ponta** (com gerador e LLM como juiz). A comparação foi com o embedding e com quatro rerankers, entre eles o **Cohere Rerank 4 Pro** e o **bge-reranker-v2-m3**.

- **Alucinação na resposta final:** com o JEV decidindo o contexto, **3 respostas alucinadas em 299**, contra **23 com o Cohere top-3** e 24 com o embedding top-3 (**~8× menos**, p < 0,001).
- **Respostas erradas dadas com confiança** (erradas e sem "não sei", o que inclui responder uma pergunta sem resposta a partir de um trecho parecido): **13 com o JEV, contra 48 com o Cohere top-3** (~3,7× menos, p < 0,001). Nas 54 perguntas sem resposta: **15% contra 69%**.
- **Respostas corretas:** 83,6% com o JEV, contra 81,9% com o Cohere top-3. **Empate estatístico:** a queda de alucinação não custou acurácia no total.
- **Custo total por pergunta** (seleção de contexto + gerador): **US$ 0,39 a cada mil perguntas com o JEV, contra US$ 2,84 com o Cohere top-3 (~7× menos)**. Só a etapa de avaliação dos trechos custa ~10× menos (US$ 0,26 contra ~US$ 2,50).
- **Recuperação:** para **ordenar** os trechos, os rerankers de ponta são ligeiramente melhores (97% contra 95%, sem significância). Para **detectar que a resposta não está nos candidatos**, o JEV é bem superior (AUC 0,95 contra 0,85 da Cohere).
- **O preço a pagar:** mais "não sei" indevidos. Nas perguntas com resposta, o JEV acerta 83%, contra 94% do Cohere top-3.

Para aplicações em que responder "não sei" é preferível a inventar, os resultados indicam que um modelo de decisão como o JEV é uma camada de controle mais adequada que os rerankers de ponta, a uma fração do custo.

---

## 1. Introdução

Um pipeline de RAG típico segue quatro passos: (1) transformar a base de documentos em vetores; (2) buscar os *k* trechos mais parecidos com a pergunta; (3) opcionalmente, reordenar esses trechos com um reranker; (4) enviar os primeiros ao LLM, que gera a resposta.

Esse desenho tem duas fragilidades:

1. **Similaridade não é o mesmo que resposta.** O embedding mede se o trecho "fala do mesmo assunto". Um trecho sobre quem projetou o Cristo Redentor é muito parecido com a pergunta "quando o Cristo foi inaugurado?", mas não a responde. No nosso teste, em **15% das perguntas que tinham um trecho útil entre os candidatos**, o 1º colocado pelo embedding não respondia.
2. **Ninguém decide que não há resposta.** O passo (4) sempre envia *k* trechos. Quando a resposta não existe na base, o LLM recebe *k* trechos plausíveis e irrelevantes e tende a "ajudar" com uma resposta inventada. Os rerankers corrigem a **ordem**, mas não mudam isso: são treinados para ordenar e usados para devolver os top-*k*.

**Tese.** Um modelo de decisão, que responde "este trecho ajuda a responder a pergunta?" com uma probabilidade comparável entre perguntas, pode fazer o que o reranker não faz: **enviar ao LLM só os trechos que servem, e nenhum quando nenhum serve.** Isso ataca a alucinação na origem, reduz o contexto enviado ao LLM e, nas perguntas sem resposta, evita até a chamada ao LLM. Testamos essa tese com o JEV contra rerankers de ponta.

---

## 2. O JEV

O JEV (`typesafe/jev-1.13`, da TypeSafe, acessado pelo OpenRouter) é um modelo de **decisão**: recebe um texto (`state`) e perguntas tipadas, e devolve probabilidades, não texto. Há três tipos de pergunta:

| Tipo | Pergunta | Saída |
|---|---|---|
| `noul` | sim ou não | probabilidade de "sim", de 0 a 1 |
| `choice` | múltipla escolha | distribuição de probabilidade entre as opções |
| `score` | escala ordinal | nota e probabilidade de cada nível |

Diferenças para um reranker:
- **A pergunta é escrita por quem usa o modelo.** Usamos: *"The passage contains information that helps answer the question"*.
- **A saída é uma probabilidade** com o mesmo sentido em qualquer pergunta. Isso permite um corte fixo.
- **A mesma chamada pode avaliar um conjunto de trechos** (*"os trechos, juntos, bastam para responder?"*) e várias perguntas sobre o mesmo texto.

---

## 3. Metodologia

### 3.1 Dados

- **Base:** os **1.204 parágrafos** do conjunto de desenvolvimento do **SQuAD 2.0** (Wikipédia em inglês), cada um com o título do artigo no início.
- **Perguntas:** uma amostra aleatória (semente 42) de 400 perguntas com pelo menos 4 palavras:
  - **100 de ajuste**, usadas só para escolher os cortes de nota;
  - **300 de teste**, com **246 com resposta e 54 sem resposta**.
- **Proporção de perguntas sem resposta:** 20%, abaixo dos ~50% do SQuAD 2.0, para ficar mais perto do tráfego real.
- **Perguntas sem resposta do SQuAD 2.0:** foram escritas por humanos para **parecer respondíveis** por um parágrafo específico. São armadilhas deliberadas.

**Gabarito.** Um trecho **responde** a uma pergunta quando é o parágrafo de origem **ou** contém uma das respostas aceitas (com 4 ou mais caracteres). A segunda regra existe porque o mesmo fato aparece em vários parágrafos do mesmo artigo. Nas perguntas sem resposta, nenhum trecho responde.

### 3.2 Recuperação

- **Embedding:** `openai/text-embedding-3-small` (1.536 dimensões).
- **Índice:** FAISS `IndexFlatIP`, com vetores normalizados (similaridade de cosseno).
- **Candidatos:** os **10 trechos** mais parecidos, os mesmos para todos os métodos e para as duas etapas.

Das 300 perguntas de teste, **239** têm ao menos um trecho que responde entre os 10 candidatos. As outras **61** não têm: são as 54 sem resposta, mais 7 cujo parágrafo certo a busca não trouxe.

### 3.3 Etapa 1: avaliação da recuperação

Cada método dá uma nota a cada um dos 10 candidatos. Nenhum LLM gerador ou juiz participa desta etapa.

| Método | Tipo | Onde roda | Observação |
|---|---|---|---|
| cosine | similaridade do embedding | junto com a busca | o que um RAG sem reranker usa |
| **JEV** | modelo de decisão | API (OpenRouter) | 10 chamadas de relevância + 1 de conjunto, em paralelo |
| MiniLM | reranker `ms-marco-MiniLM-L-12-v2` | local (CPU) | pequeno, clássico |
| BGE | reranker `bge-reranker-base` | local (CPU) | médio |
| **bge-v2-m3** | reranker `BAAI/bge-reranker-v2-m3` | local (CPU) | **aberto, de ponta** |
| **Cohere** | `rerank-v4.0-pro` | API (Cohere) | **comercial, de ponta** |

**Métricas:**
- **Ordenação** (nas 239 perguntas com trecho útil): acerto do 1º lugar, MRR, e armadilhas corrigidas (casos em que o 1º do embedding não responde, mas outro candidato responde).
- **Decisão com corte fixo:** o corte de cada método é o de maior F1 nas perguntas de **ajuste**, aplicado sem alteração nas de **teste**. A partir dele: precisão, recall, perguntas servidas (algum trecho útil enviado) e **abstenção correta** (nenhum trecho enviado quando nenhum serve).
- **Separação:** AUC por trecho (útil contra inútil, entre perguntas diferentes).
- **Pontos de operação:** a abstenção de cada método quando o corte é ajustado para servir 90%, 85% ou 80% das perguntas.
- **Detecção de ausência de resposta:** AUC por pergunta da maior nota de cada método.

### 3.4 Etapa 2: avaliação ponta a ponta

Mesmos candidatos e mesmas notas da etapa 1. Quatro **políticas de contexto** decidem o que chega ao gerador:

| Política | O que envia ao gerador |
|---|---|
| **top3** | os 3 trechos mais parecidos pelo embedding (prática comum) |
| **cohere_top3** | os 3 melhores segundo a Cohere (prática comum com reranker) |
| **cohere_cut** | todos os trechos com nota Cohere ≥ 0,891 |
| **jev_cut** | todos os trechos com relevância JEV ≥ 0,97 |

- **Cortes:** escolhidos nas perguntas de ajuste, como na etapa 1.
- **Sem trecho aprovado:** quando uma política não mantém nenhum trecho, **o gerador não é chamado** e o sistema devolve uma mensagem fixa ("No relevant passage was found to answer the question.").
- **Gerador:** `openai/gpt-4.1-mini`, instruído a responder só com o contexto, citar os trechos usados e dizer que não sabe quando o contexto não basta.
- **Juiz:** `anthropic/claude-sonnet-5`, com temperatura 0. Avalia cada resposta contra a resposta de referência do SQuAD.

**Categorias do juiz:**

- **Correta:** todas as partes da pergunta respondidas corretamente. Numa pergunta sem resposta, a única resposta correta é "não sei".
- **Alucinação:** a resposta afirma algo que não está no contexto recebido pelo gerador, ou que contradiz a referência.
- **Errada com confiança:** a resposta não é correta e não diz "não sei". Isso inclui as alucinações e também **responder uma pergunta sem resposta a partir de um trecho parecido**. O juiz não marca esse segundo caso como alucinação, porque a afirmação está no contexto, mas para o usuário ela é igualmente errada.
- **"Não sei" incorreto:** a resposta diz que não sabe, mas não é correta. É uma abstenção indevida (a resposta existia) ou, em poucos casos, uma abstenção misturada com uma afirmação sem apoio (2 casos no cohere_top3 e 2 no cohere_cut, nas perguntas sem resposta).

Toda resposta cai em exatamente um de três grupos: correta, errada com confiança ou "não sei" incorreto. As alucinações são um subconjunto das erradas com confiança.

**Dois conjuntos de "sem resposta":**

- **54 perguntas sem resposta:** a resposta não está na base.
- **61 perguntas sem trecho útil entre os candidatos:** as 54 acima, mais 7 em que a resposta está na base, mas a busca não a trouxe. A etapa 1 usa este conjunto.

**Pergunta excluída:** o filtro de segurança do juiz se recusou a avaliar uma pergunta **com** resposta (sobre evasão imune). Ela foi excluída de todas as políticas, e a etapa 2 fica com **299 perguntas: 245 com resposta e 54 sem resposta**.

**Estatística:**
- **Diferenças pareadas:** teste exato de McNemar.
- **Intervalos de confiança de 95%:** *bootstrap* sobre as perguntas, com 2.000 a 5.000 reamostragens.

**Custo e latência:**
- **Custo do JEV e do gerador:** o `usage.cost` real devolvido pelo OpenRouter.
- **Custo da Cohere:** preço de tabela.
- **Latência da Cohere:** medida à parte, em 50 chamadas sem espera entre elas.

---

## 4. Resultados

### 4.1 Resultado principal: alucinação na resposta final

Resultados em **299 perguntas**. Uma foi excluída porque o filtro de segurança do juiz se recusou a avaliá-la (seção 6).

| Métrica | top3 | cohere_top3 | cohere_cut | **jev_cut** |
|---|---|---|---|---|
| **Respostas alucinadas** | 24 (8,0%) | 23 (7,7%) | 7 (2,3%) | **3 (1,0%)** |
| **Respostas erradas dadas com confiança** | 46 (15,4%) | 48 (16,1%) | 24 (8,0%) | **13 (4,3%)** |
| …nas 54 perguntas sem resposta | 35 (65%) | 37 (69%) | 17 (31%) | **8 (15%)** |
| Respostas corretas | 81,3% | 81,9% | 79,3% | **83,6%** |
| Corretas, perguntas com resposta (n = 245) | 91% | **94%** | 82% | 83% |
| **Corretas, perguntas sem resposta** (n = 54) | 35% | 28% | 65% | **85%** |
| **Alucinação nas perguntas sem resposta** | 35% | 30% | 7% | **4%** |
| "Não sei" indevido (perguntas com resposta) | 4% | **2%** | 15% | 15% |
| Perguntas sem chamada ao gerador | 0% | 0% | 22% | **27%** |

**Comparação pareada do jev_cut com cada política:**

| Contra | Métrica | jev_cut melhor | outro melhor | McNemar p | Diferença jev_cut − outro (IC 95%) |
|---|---|---|---|---|---|
| top3 | alucinação | 22 | 1 | **< 0,001** | **−7,0 pts** (−10,0 a −4,0) |
| cohere_top3 | alucinação | 21 | 1 | **< 0,001** | **−6,7 pts** (−10,0 a −4,0) |
| cohere_cut | alucinação | 6 | 2 | 0,29 | −1,3 pts (−3,3 a +0,3) |
| top3 | erradas com confiança | 36 | 3 | **< 0,001** | **−11,0 pts** (−15,1 a −7,4) |
| cohere_top3 | erradas com confiança | 38 | 3 | **< 0,001** | **−11,7 pts** (−15,7 a −7,7) |
| cohere_cut | erradas com confiança | 19 | 8 | 0,05 | −3,7 pts (−7,0 a −0,3) |
| top3 | corretas | 33 | 26 | 0,44 | +2,3 pts (−2,7 a +7,4) |
| cohere_top3 | corretas | 34 | 29 | 0,62 | +1,7 pts (−3,3 a +7,0) |
| cohere_cut | corretas | 38 | 25 | 0,13 | +4,3 pts (−0,7 a +9,7) |

**Leitura:**
- **Contra a prática comum (top-3, com ou sem Cohere):** o JEV reduz a alucinação de ~8% para 1%, com alta significância, sem perder acurácia no total.
- **Contra a prática comum, nas erradas com confiança:** de 48 (Cohere top-3) para 13, ~3,7× menos (p < 0,001).
- **Contra a Cohere com corte:** uma configuração menos comum. A diferença de alucinação (7 contra 3) e de acurácia (79% contra 84%) favorece o JEV, mas **não é estatisticamente significativa**. Nas erradas com confiança (24 contra 13), a diferença fica **no limite** (p = 0,05; IC 95% de −7,0 a −0,3 pontos).
- **A troca:** nas perguntas **com** resposta, as políticas com corte dizem "não sei" indevidamente em 15% dos casos, contra 2 a 4% do top-3. O JEV troca uma parte das respostas certas por muito menos respostas inventadas. No total, a troca compensa (83,6% contra 81,9%), mas ela precisa ser declarada.
- **Onde está o ganho:** nas perguntas sem resposta, 85% das respostas do JEV foram corretas ("não sei"), contra 28% do Cohere top-3. Na maior parte desses casos, o gerador nem foi chamado.

**Por que as políticas erram:** quando o contexto tinha um trecho útil, o gerador acertou 97 a 99% das vezes em todas as políticas. **Os erros vêm do contexto, não do gerador.** O top-3 inclui um trecho útil em 92 a 96% das perguntas com resposta, e as políticas com corte em 84%.

### 4.2 Custo e latência do pipeline

Custo a cada mil perguntas:

| Custo | top3 | cohere_top3 | cohere_cut | **jev_cut** |
|---|---|---|---|---|
| Tokens de entrada do gerador (média por pergunta, medido) | 605 | 602 | 222 | **200** |
| Gerador (medido) | US$ 0,341 | US$ 0,341 | US$ 0,150 | **US$ 0,133** |
| Seleção de contexto | 0 | US$ 2,500¹ | US$ 2,500¹ | **US$ 0,256** (medido) |
| **Total** | US$ 0,341 | US$ 2,841 | US$ 2,650 | **US$ 0,389** |

¹ Preço de tabela da Cohere (~US$ 2,50 a cada mil buscas, segundo fontes de terceiros).

- **Contra o Cohere top-3:** **~7× mais barato no total**, e ~10× mais barato só na seleção de contexto.
- **Contra o top-3 sem reranker:** **+14%** de custo total (US$ 0,39 contra US$ 0,34) para cortar a alucinação de 8% para 1%.
- **Com um gerador mais caro,** a vantagem cresce: o JEV envia **67% menos tokens** ao gerador que o top-3 e dispensa o gerador em 27% das perguntas. A média de 200 tokens inclui as perguntas em que o gerador não é chamado; quando ele é chamado, a média é 276.
- **As 11 chamadas do JEV:** o custo e a latência medidos incluem 11 chamadas por pergunta, em paralelo: 10 de relevância (uma por trecho) e 1 sobre o conjunto, que a política jev_cut não usa. O custo real da política seria ~9% menor. A Cohere avalia os 10 trechos numa única chamada, então o JEV faz ~10× mais requisições, o que pesa em limites de requisições.

| Latência | p50 | p95 | Observação |
|---|---|---|---|
| JEV, 10 candidatos | 0,70 s | 1,62 s | 11 chamadas em paralelo |
| Cohere, 10 candidatos | **0,54 s** | 1,75 s | 1 chamada (medida à parte, 50 chamadas) |
| Gerador (top3 / jev_cut) | 2,10 s / 1,89 s | – | o jev_cut inclui 0 s quando o gerador não é chamado |

A latência da seleção é equivalente: a Cohere é ~0,16 s mais rápida na mediana, e os dois empatam no p95. Como o gerador domina o tempo total, a diferença entre os pipelines é pequena.

### 4.3 Etapa 1: por que isso acontece (recuperação)

**As armadilhas semânticas existem.** Em **36 das 239 perguntas (15%)**, o trecho mais parecido pelo embedding **não responde**, mas outro candidato responde.

**Ordenar: os rerankers de ponta estão ligeiramente à frente.**

| Método | 1º lugar responde | MRR | Armadilhas corrigidas (de 36) |
|---|---|---|---|
| cosine (embedding) | 85% | 0,90 | – |
| MiniLM | 93% | 0,96 | 25 |
| BGE | 89% | 0,93 | 26 |
| bge-v2-m3 | **97%** | **0,98** | 30 |
| Cohere | **97%** | **0,98** | **32** |
| **JEV** | 95% | 0,97 | 29 |

Contra o bge-v2-m3 e a Cohere, o JEV fica 2 a 3 pontos atrás, sem significância (p = 0,30 e p = 0,09). Contra o embedding e o BGE-base, fica à frente com significância (+10 e +5 pontos).

**Decidir com corte fixo: o JEV lidera.** Corte escolhido nas perguntas de ajuste:

| Método | AUC por trecho | Precisão | Perguntas servidas | **Abstenção correta** |
|---|---|---|---|---|
| cosine | 0,727 | 44% | 55% | 75% |
| MiniLM | 0,801 | 73% | 69% | 57% |
| BGE | 0,824 | 62% | 77% | 44% |
| bge-v2-m3 | 0,838 | 72% | 87% | 51% |
| Cohere | 0,850 | 80% | 87% | 59% |
| **JEV** | **0,868** | **84%** | 86% | **87%** |

A diferença de abstenção do JEV é de **+28 pontos sobre a Cohere** (IC 95%: +13 a +43) e **+36 sobre o bge-v2-m3** (+21 a +49). No uso comum dos rerankers (sempre os top-*k*), a abstenção é **0% por construção**. Os 59% da Cohere nesta tabela vêm do corte ajustado. Na tabela seguinte, com o corte ajustado para servir 85% das perguntas, ela chega a 61%.

**Pontos de operação: a vantagem não depende do corte.**

| Perguntas servidas (alvo) | cosine | MiniLM | BGE | bge-v2-m3 | Cohere | **JEV** |
|---|---|---|---|---|---|---|
| ≥ 90% | 18% | 28% | 28% | 44% | 51% | **75%** |
| ≥ 85% | 31% | 34% | 34% | 52% | 61% | **87%** |
| ≥ 80% | 34% | 43% | 38% | 54% | 69% | **87%**¹ |

¹ As notas do JEV se concentram perto de 0 e de 1. O primeiro corte que atinge ≥ 80% já serve 86%.

**Detectar que a resposta não está nos candidatos** (AUC por pergunta):

| Sinal | AUC |
|---|---|
| maior nota do cosine | 0,690 |
| maior nota do MiniLM | 0,704 |
| maior nota do BGE | 0,723 |
| maior nota do bge-v2-m3 | 0,789 |
| maior nota da Cohere | 0,846 |
| **maior nota do JEV** | **0,946** |
| JEV, pergunta sobre o conjunto (top-5) | 0,895 |

A diferença do JEV é de +0,10 sobre a Cohere (IC 95%: +0,05 a +0,15). Um resultado negativo: a pergunta sobre o conjunto não superou a maior nota individual. No SQuAD, em que quase toda pergunta se resolve com um parágrafo, essa capacidade não fez falta.

---

## 5. Discussão

### 5.1 Por que o JEV decide melhor, se ordena um pouco pior?

Os rerankers são treinados para **ordenar**: a nota deles só precisa ser comparável **dentro da mesma pergunta**, e a escala muda de uma pergunta para outra. Um corte fixo na nota de um reranker deixa passar lixo numas perguntas e barra trechos bons noutras. O JEV responde a uma pergunta de **sim ou não** com uma probabilidade, e, neste teste, **as notas dele foram comparáveis entre perguntas**: o corte ajustado nas perguntas de ajuste funcionou bem nas de teste. Não medimos a calibração em si (se 0,97 corresponde a 97% de acerto). Os cortes do JEV e da Cohere foram ajustados da mesma forma. A etapa 1 mostra isso na separação (AUC 0,946 contra 0,846 na detecção por pergunta), e a etapa 2 mostra a consequência: **menos contexto inútil chega ao gerador, e ele inventa menos.**

### 5.2 Arquitetura recomendada

1. **Recuperar** candidatos com embeddings, como de costume.
2. **Avaliar** os candidatos com o JEV, em paralelo (~0,7 s para 10).
3. **Se nenhum passar do corte:** não chamar o LLM e devolver uma mensagem fixa, como *"Não encontrei essa informação nos documentos consultados."*. Isso é mais confiável e mais barato que instruir o LLM a dizer que não sabe.
4. **Se algum passar:** enviar ao LLM **só** os trechos aprovados.

**Ajustes e variações:**
- **Calibrar o corte** para o risco da aplicação: um corte mais baixo reduz os "não sei" indevidos e aumenta o risco de alucinação.
- **Buscar de novo antes de desistir:** quando nada passa, reformular a pergunta e fazer uma segunda busca. Isso ataca os casos em que a resposta existe, mas a busca não a trouxe.
- **Ordenar melhor:** usar um reranker de ponta para ordenar e o JEV para decidir, pagando os dois.

### 5.3 Onde a tese se aplica

Em aplicações em que **responder errado custa mais que responder "não sei"** (jurídico, saúde, suporte técnico, conformidade), o JEV é a escolha indicada por estes resultados. Onde quase sempre há resposta na base e um "não sei" indevido é caro, o top-3 com um reranker de ponta acerta mais nas perguntas com resposta (94% contra 83%), ao custo de mais alucinações.

---

## 6. Fragilidades e cuidados

1. **Uma base só, em um domínio só.** O SQuAD é Wikipédia em inglês, com parágrafos curtos (~205 tokens). O resultado pode mudar em domínios técnicos, em outros idiomas e com trechos maiores.
2. **Uma execução.** O JEV, o gerador e o juiz têm variação entre chamadas, e não repetimos o experimento.
3. **Poucas perguntas sem resposta:** 54. A redução de alucinação contra o top-3 é muito significativa, mas a comparação com a Cohere com corte (7 contra 3 alucinações) não é.
4. **O juiz é um LLM.** A alucinação é a marcação dele, sem revisão humana. Uma amostra revisada por humanos aumentaria a confiança.
5. **Uma pergunta excluída:** o filtro de segurança do juiz (`claude-sonnet-5`) se recusou a avaliar uma pergunta de biologia sobre evasão imune, que tinha resposta. Ela ficou fora da análise para todas as políticas.
6. **A métrica de alucinação do juiz subconta o problema:** responder uma pergunta sem resposta a partir de um trecho parecido não é marcado como alucinação, porque está no contexto. Por isso o artigo também reporta as respostas erradas dadas com confiança.
7. **"Não sei" indevido:** as políticas com corte (JEV e Cohere) deixam sem resposta 15% das perguntas que tinham resposta. É uma troca real, que depende do corte.
8. **"Não está nos candidatos" não é "não está na base".** Das 61 perguntas sem trecho útil entre os candidatos, 7 tinham resposta na base. Uma mensagem como *"não encontrei nos documentos consultados"* é mais honesta que *"não está nos documentos"*.
9. **O preço da Cohere vem de fontes de terceiros.** A página oficial só mostra o preço de instâncias dedicadas. O "~7× mais barato" depende desse preço.
10. **O custo do JEV inclui a pergunta sobre o conjunto** (11 chamadas por pergunta), que a política jev_cut não usa. O custo real da política seria ~9% menor.
11. **O gabarito por texto é aproximado.** Contar como útil um trecho que contém a resposta pode errar em casos raros.
12. **A latência dos rerankers locais foi medida sem GPU** e não representa produção. A comparação de latência válida é JEV contra Cohere, ambos por API.
13. **A instrução do JEV é próxima do critério do gabarito** ("ajuda a responder"). É o uso natural do modelo, mas uma instrução diferente poderia dar outros resultados.
14. **Outros rerankers ficaram de fora:** Qwen3-Reranker, Voyage e Jina não foram avaliados.
15. **Conflito de interesse:** nenhum. O autor não tem vínculo com a TypeSafe, a Cohere, a Anthropic, a OpenAI ou o OpenRouter, nem recebeu financiamento ou créditos deles. Todo o uso de API foi pago pelo autor (a Cohere, pela chave de teste gratuita).

---

## 7. Próximos passos

1. **Outros domínios:** coleções do BEIR (SciFact, FiQA, NFCorpus) e documentos reais de um domínio de interesse, inclusive em português.
2. **Repetições:** 3 execuções, para medir a variação, e mais perguntas sem resposta.
3. **Revisão humana** de uma amostra das marcações de alucinação do juiz.
4. **Segunda busca antes de desistir**, para recuperar parte dos "não sei" indevidos.
5. **JEV em lote:** uma única chamada com todos os trechos, para reduzir custo e número de chamadas.
6. **Verificação da resposta:** usar o JEV depois da geração ("a resposta está apoiada nos trechos?") como segunda camada.
7. **A combinação "Cohere ordena e JEV decide"** e a política do JEV com cortes diferentes (curva de alucinação contra "não sei" indevido).

---

## 8. Conclusão

No SQuAD 2.0, a prática comum de enviar ao LLM os 3 trechos mais relevantes produziu respostas alucinadas em ~8% das perguntas, e em 30 a 35% das perguntas sem resposta, **com ou sem um reranker de ponta**. Os rerankers ordenam bem (97% de acerto no 1º lugar), mas não sabem dizer que não há nada a enviar.

Deixar o JEV decidir o contexto, com um corte fixo na probabilidade de relevância e sem chamar o LLM quando nada passa:
- **reduziu as alucinações de 23 para 3** em 299 perguntas, contra o Cohere top-3 (p < 0,001);
- **reduziu as respostas erradas dadas com confiança de 48 para 13** (p < 0,001);
- **manteve a acurácia total** (83,6% contra 81,9%);
- **custou ~7× menos** que o pipeline com a Cohere;
- **teve latência equivalente.**

A contrapartida é mais "não sei" nas perguntas que tinham resposta. Para aplicações em que evitar alucinação é prioridade, os resultados indicam que um modelo de decisão como o JEV é uma camada de controle mais adequada, e mais barata, que os rerankers de ponta.

---

## Como este trabalho foi feito

Este estudo foi construído junto com um assistente de IA, e vale dizer com precisão quem fez o quê.

- **A pergunta e a tese são do autor.** O ponto de partida foi uma dúvida prática: um modelo de decisão sim/não separa melhor que um reranker os trechos que respondem à pergunta dos que só parecem parecidos? As afirmações deste artigo são as que o autor decidiu fazer.
- **O código foi escrito pelo Claude Opus 5.5** (Anthropic), no Claude Code, a partir das instruções do autor: o pipeline de RAG, os experimentos, a estatística e as figuras.
- **O método foi construído em diálogo.** Várias escolhas de método foram propostas pela IA e aceitas, alteradas ou recusadas pelo autor. Entre elas: os controles de corte por similaridade e de top-*k*, a comparação com rerankers de ponta, o ajuste dos cortes num conjunto separado de perguntas e a medição do custo real. Outras escolhas partiram do autor, como colocar o portão na frente de um RAG top-1, o foco em alucinação e custo, e quais rerankers testar e em que ordem.
- **O texto foi redigido com o Claude Opus 5.5** e revisado e editado pelo autor.
- **Todos os números vêm do código e dos resultados publicados**, não do texto da IA. Qualquer pessoa pode refazer os experimentos a partir do repositório.
- **Uma nota sobre fornecedores:** o código foi escrito, e as respostas avaliadas, por modelos da Anthropic (Claude Opus 5.5 e `claude-sonnet-5`). Nenhum produto da Anthropic estava entre os sistemas comparados.

---

## Apêndice A: Reprodutibilidade

**Código** (neste repositório):

| Arquivo | Função |
|---|---|
| `prepare_squad.py` | baixa o SQuAD 2.0 e gera a base e as perguntas (semente 42) |
| `rerank_experiment.py` | etapa 1: notas de todos os métodos, métricas e relatório |
| `e2e_experiment.py` | etapa 2: políticas de contexto, gerador, juiz e relatório |
| `gate_experiment.py` | experimento exploratório do portão (apêndice B) |
| `rag.py`, `jev.py`, `meter.py` | pipeline de RAG, cliente do JEV e medidor de custo real |

**Comandos:**

```bash
python prepare_squad.py
python rerank_experiment.py                      # embedding, JEV, MiniLM e BGE
python rerank_experiment.py --add bge_m3         # bge-reranker-v2-m3 (local)
python rerank_experiment.py --add cohere         # Cohere rerank-v4.0-pro (COHERE_API_KEY no .env)
python rerank_experiment.py --cohere-latency 50  # latência da Cohere sem limite de requisições
python e2e_experiment.py                         # etapa 2 (gerador e juiz)
```

**Resultados:** `results/rerank_report.md`, `results/rerank_results.json`, `results/e2e_report.md` e `results/e2e_results.json`.

**Instruções do JEV:**
- relevância: *"The passage contains information that helps answer the question"*
- conjunto: *"The passages, together, contain all the information needed to fully answer the question, including all of its parts"*

**Versões:** JEV `typesafe/jev-1.13` (resposta `jev-1.13-20260917`); `openai/text-embedding-3-small`; `openai/gpt-4.1-mini`; `anthropic/claude-sonnet-5`; Cohere `rerank-v4.0-pro`; `BAAI/bge-reranker-v2-m3` e `BAAI/bge-reranker-base`; `Xenova/ms-marco-MiniLM-L-12-v2`; FAISS 1.15; sentence-transformers 6.1; PyTorch 2.14 (CPU); fastembed 0.8.1; Python 3.14.

**Custo total do projeto:** US$ 6,86 no OpenRouter, somando todos os experimentos e testes de desenvolvimento. A etapa 1 custou ~US$ 0,10 com o JEV, e a etapa 2 ~US$ 5,30 (quase tudo com o juiz). A Cohere foi usada na cota gratuita de teste.

## Apêndice B: Experimentos exploratórios

Antes do SQuAD, a ideia foi desenvolvida numa base própria em português, com 20 trechos e 16 perguntas, cheia de distratores do mesmo assunto:

- **Filtro de relevância:** os distratores do mesmo tema tiveram similaridade de 0,54 a 0,61, e o JEV deu a eles notas de 0,03 a 0,10.
- **Portão na frente de um RAG top-1:** as respostas corretas subiram de 56% para 94% (McNemar p = 0,03), e o portão custou +0,5 s quando aprovou o 1º trecho. Um top-2 fixo teve a mesma acurácia, o que motivou a comparação com os top-*k* e com os rerankers.
- **Portão no SQuAD (30 perguntas):** 83% de respostas corretas, contra 73% do top-1 e 80% do top-3. O portão aprovou o 1º trecho em 60% das perguntas, com +0,6 s.

## Apêndice C: Fontes do preço da Cohere

- Cohere Pricing (página oficial; preços de instâncias dedicadas e definição de busca): https://cohere.com/pricing
- Cohere API Pricing, Puter: https://developer.puter.com/tutorials/cohere-api-pricing/
- Cohere API Pricing 2026, AI Pricing Guru: https://www.aipricing.guru/cohere-pricing/
