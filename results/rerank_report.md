# Experimento: o JEV distingue "responde" de "só parece"?

Gerado em 2026-09-24 22:28. Base: `data/squad/corpus.jsonl` (1204 trechos). Perguntas de teste: 300 (246 com resposta, 54 sem resposta); perguntas de ajuste (só para escolher os cortes): 100. O embedding traz 10 candidatos por pergunta; cada candidato recebe uma nota de cada método.

- **cosine:** similaridade do embedding (`openai/text-embedding-3-small`), o que um RAG comum usa.
- **jev:** relevância do JEV (`typesafe/jev-1.13`): "o trecho contém informação que ajuda a responder a pergunta".
- **minilm:** reranker `Xenova/ms-marco-MiniLM-L-12-v2` (local).
- **bge:** reranker `BAAI/bge-reranker-base` (local).
- **bge_m3:** reranker `BAAI/bge-reranker-v2-m3` (local).
- **cohere:** reranker `cohere/rerank-v4.0-pro` (local).

Um trecho "responde" quando é o trecho gold ou contém uma das respostas aceitas.

## 1. Quem põe em 1º lugar um trecho que responde?

Considerando as 239 perguntas de teste em que algum dos 10 candidatos responde (nas outras 61, nenhum método pode acertar).

| Método | 1º lugar responde | MRR | Armadilhas corrigidas |
|---|---|---|---|
| cosine | 85% | 0.90 | - |
| jev | 95% | 0.97 | 29/36 |
| minilm | 93% | 0.96 | 25/36 |
| bge | 89% | 0.93 | 26/36 |
| bge_m3 | 97% | 0.98 | 30/36 |
| cohere | 97% | 0.98 | 32/36 |

**Armadilhas:** perguntas em que o trecho mais parecido (1º do cosine) **não** responde, mas outro candidato responde. Foram 36 de 239 (15%). A coluna mostra em quantas delas o método pôs um trecho que responde em 1º lugar.

**MRR:** média de 1/posição do primeiro trecho que responde (1 = sempre em 1º; 0.5 = em média em 2º).

### JEV contra cada método (1º lugar responde, pareado)

| Contra | JEV acerta e o outro erra | O outro acerta e JEV erra | McNemar p | Diferença (IC 95%) |
|---|---|---|---|---|
| cosine | 29 | 6 | 0.000 | +10% (+5% a +14%) |
| minilm | 12 | 9 | 0.664 | +1% (-3% a +5%) |
| bge | 22 | 9 | 0.029 | +5% (+1% a +10%) |
| bge_m3 | 5 | 10 | 0.302 | -2% (-5% a +1%) |
| cohere | 3 | 10 | 0.092 | -3% (-6% a +0%) |

## 2. A nota separa "responde" de "só parece" com um corte fixo?

Na prática, um corte fixo decide quais trechos vão para o LLM. Aqui o corte de cada método foi escolhido nas perguntas de ajuste (maior F1) e aplicado nas de teste.

| Método | Separação (AUC) | Corte | Precisão | Recall | Trechos mantidos por pergunta | Perguntas com um trecho que responde mantido | Perguntas sem resposta em que nada passou |
|---|---|---|---|---|---|---|---|
| cosine | 0.727 | 0.596 | 44% | 39% | 1.25 | 55% | 75% |
| jev | 0.868 | 0.97 | 84% | 52% | 0.85 | 86% | 87% |
| minilm | 0.801 | 4.95 | 73% | 42% | 0.81 | 69% | 57% |
| bge | 0.824 | 0.481 | 62% | 49% | 1.11 | 77% | 44% |
| bge_m3 | 0.838 | 0.788 | 72% | 54% | 1.04 | 87% | 51% |
| cohere | 0.850 | 0.891 | 80% | 54% | 0.95 | 87% | 59% |

**AUC:** chance de um trecho que responde ter nota maior que um que não responde, comparando trechos de perguntas diferentes (0.5 = sorte; 1.0 = separação perfeita). É o que importa para um corte fixo funcionar em qualquer pergunta.

## 3. O conjunto de candidatos tem a resposta? (o que só o JEV faz)

Antes de chamar o LLM, dá para saber se a resposta está entre os candidatos? Das 300 perguntas de teste, 239 têm a resposta entre os 10 candidatos e 61 não têm. Os rerankers só podem usar a maior nota individual; o JEV também pode avaliar o conjunto ("os 5 primeiros trechos, juntos, respondem a pergunta?").

| Sinal | Separação (AUC) |
|---|---|
| maior nota do cosine | 0.690 |
| maior nota do jev | 0.946 |
| maior nota do minilm | 0.704 |
| maior nota do bge | 0.723 |
| maior nota do bge_m3 | 0.789 |
| maior nota do cohere | 0.846 |
| JEV, suficiência do conjunto (top-5) | 0.895 |

Para a suficiência do conjunto, "tem a resposta" significa que algum dos 5 primeiros candidatos responde.

## 4. Custo e latência para avaliar os candidatos

| Método | Latência p50 por pergunta (s) | Latência p95 (s) | Custo por pergunta | Onde roda |
|---|---|---|---|---|
| cosine | 0 (já calculado na busca) | 0 | 0 | junto com a busca |
| jev | 0.70 | 1.62 | US$ 0.000256 (medido) | API, 11 chamadas em paralelo |
| minilm | 0.55 | 1.21 | 0 (CPU local) | este computador, sem GPU |
| bge | 3.40 | 5.86 | 0 (CPU local) | este computador, sem GPU |
| bge_m3 | 15.06 | 26.61 | 0 (CPU local) | este computador, sem GPU |
| cohere | 0.54 | 1.75 (medida à parte, 50 chamadas) | cobrado por busca (aqui, cota de teste gratuita) | API, 1 chamada com os 10 trechos |

Custo do JEV medido pelo `usage.cost` do OpenRouter; 0 chamadas vieram sem custo informado. Por mil perguntas: US$ 0.26.

## Limitações

- Uma base (SQuAD 2.0, Wikipédia em inglês) e uma execução. O comportamento pode mudar em outros domínios.
- Os rerankers leem no máximo ~512 tokens por par; parágrafos maiores são truncados.
- A latência dos rerankers depende da CPU desta máquina; com GPU, cairia bastante. A do JEV depende da rede.
- A verdade usa o trecho gold ou a presença de uma resposta aceita no texto, o que pode errar em casos raros.
