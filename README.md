# better_rag: RAG com filtro de relevância via JEV

> **Artigo:** [JEV Gating vs. Top-k Reranking: 8× Fewer RAG Hallucinations at One-Seventh the Cost](artigo_medium_en.md) (inglês) · [versão em português](artigo_rascunho.md)
>
> **Experimentos principais (SQuAD 2.0, 300 perguntas):**
> - `rerank_experiment.py`: JEV contra embedding e rerankers (MiniLM, BGE, bge-reranker-v2-m3, Cohere Rerank 4 Pro), só na recuperação. Relatório: [results/rerank_report.md](results/rerank_report.md)
> - `e2e_experiment.py`: quatro políticas de contexto (top3, Cohere top3, Cohere com corte, JEV com corte), com gerador e LLM como juiz. Relatório: [results/e2e_report.md](results/e2e_report.md)
> - `prepare_squad.py` gera `data/squad/` a partir do SQuAD 2.0 (CC BY-SA 4.0, Rajpurkar et al.); `make_figures.py` gera as figuras em `figures/`.
>
> Requisitos: `pip install -r requirements.txt` e, no `.env`, `apikey=<chave do OpenRouter>` (e `COHERE_API_KEY=` para a Cohere). O restante deste README documenta as fases exploratórias do projeto.

Um RAG de teste que acrescenta uma segunda camada de filtragem sobre a busca vetorial. Primeiro, a busca por embeddings (FAISS) encontra os candidatos. Depois, o **JEV** (TypeSafe, via OpenRouter) decide quais deles realmente ajudam a responder a pergunta. Só esses vão para o LLM.

Como o JEV também informa se o que foi encontrado **basta** para responder, a busca é **iterativa**: se falta informação, o sistema reformula a pergunta e busca de novo, até um limite de rodadas, antes de responder "não sei".

## A ideia

A busca vetorial mede **semelhança de assunto**, não **utilidade para a pergunta**. Um trecho sobre quem projetou o Cristo Redentor é muito parecido com a pergunta "quando o Cristo foi inaugurado?", mas não a responde.

O JEV é um modelo de decisão: ele não gera texto, apenas devolve probabilidades estruturadas. Ele é usado em dois pontos:

1. **Relevância, trecho a trecho:** *"este trecho contém informação que ajuda a responder a pergunta?"* Ficam todos os trechos com nota acima de um limiar, sem um top-N fixo.
2. **Suficiência, sobre o conjunto:** *"os trechos mantidos, juntos, respondem a pergunta por completo?"* Se não respondem, começa uma nova rodada de busca.

## Estrutura

```text
better_rag/
├── .env           # apikey=<chave do OpenRouter> (usada para tudo)
├── data.jsonl     # base de conhecimento: 20 trechos {"id", "text"}
├── jev.py         # cliente do JEV: decide(), noul(), choice(), score()
├── test_jev.py    # testes ao vivo das três funções do JEV
├── rag.py         # pipeline: JSONL -> FAISS -> JEV -> (nova rodada?) -> LLM; também o portão ask_gate()
├── eval.py / eval_questions.json   # avaliações anteriores com LLM como juiz (16 perguntas)
├── gate_experiment.py              # experimento do JEV como portão (veja a seção abaixo)
├── prepare_squad.py                # baixa e converte o SQuAD 2.0 para o formato do projeto
├── data/squad/                     # corpus e perguntas do SQuAD convertidos
└── results/                        # relatórios (.md) e dados brutos (.json) de cada experimento
```

### `jev.py`

Cliente mínimo do JEV. Usa só a biblioteca padrão do Python.

- Endpoint: `POST https://openrouter.ai/api/alpha/decisions`. O `/chat/completions` não aceita esse modelo.
- Modelo: `typesafe/jev-1.13`
- A chave vem de `OPENROUTER_API_KEY` ou do `apikey` no `.env`.

| Função | Tipo de pergunta | Retorno |
|---|---|---|
| `noul(texto, pergunta)` | sim/não | `float` de 0 a 1 (1 = sim) |
| `choice(texto, pergunta, {opção: descrição})` | múltipla escolha | `choice`, `probabilities`, `confidence` |
| `score(texto, pergunta, [nível0, nível1, ...])` | escala ordinal | `score` (média ponderada, ex.: 1.98), `probabilities`, `confidence` |
| `decide(texto, perguntas)` | chamada crua, várias perguntas | resposta completa da API |

### `rag.py`

| Componente | Escolha |
|---|---|
| Embeddings | `openai/text-embedding-3-small` (1536 dimensões), via OpenRouter |
| Índice | `faiss.IndexFlatIP` em memória, com vetores normalizados (produto interno = cosseno) |
| Relevância | `typesafe/jev-1.13`, um `noul` por trecho, até 20 chamadas em paralelo |
| Suficiência | `typesafe/jev-1.13`, um `noul` sobre o conjunto dos trechos mantidos |
| Novas consultas | `openai/gpt-4.1-mini` sugere até 3 consultas voltadas ao que ainda falta |
| Geração | `openai/gpt-4.1-mini`, responde só com o contexto e cita os ids dos trechos |

| Método | O que faz |
|---|---|
| `search(consulta, exclude)` | vizinhos mais próximos no FAISS, pulando trechos já vistos |
| `rerank(pergunta, candidatos)` | nota de relevância do JEV para cada candidato, em paralelo |
| `sufficiency(pergunta, mantidos)` | nota do JEV de que os trechos mantidos respondem a pergunta por completo |
| `next_queries(pergunta, mantidos, feitas)` | LLM gera consultas novas para as partes ainda sem resposta |
| `answer(pergunta, mantidos)` | resposta final com citações |
| `ask(pergunta)` | orquestra o laço de rodadas |

## Como funciona

```text
                     data.jsonl (20 trechos)
                              │ embed
                              ▼
                   FAISS em memória (IndexFlatIP)
                              │
 ┌──────────────────────────► Rodada N
 │                            │
 │   consultas da rodada ─────┤ 1. busca por similaridade
 │   (rodada 1: a pergunta    │    TOP_K = 5 por consulta, sem repetir trechos já vistos
 │    original)               ▼
 │                            2. JEV relevância, um noul por trecho, em paralelo
 │                            │    sempre julgado contra a pergunta ORIGINAL
 │                            │    mantém todos com nota >= MIN_RELEVANCE (0.3)
 │                            ▼
 │                            3. JEV suficiência sobre todos os trechos mantidos
 │                            │    "os trechos, juntos, respondem a pergunta por completo?"
 │                            │
 │              >= 0.5 ───────┼──────► 5. LLM responde com os trechos mantidos
 │                            │           (nenhum mantido: "Nenhum trecho relevante encontrado")
 │               < 0.5        │
 │                            ▼
 │                            4. LLM gera até 3 novas consultas
 │                            │    (reformulações e partes ainda sem resposta)
 └────────────────────────────┘    até MAX_ROUNDS = 3 ou até esgotar a base
```

Configurações no topo do `rag.py`:

```python
TOP_K = 5              # candidatos por consulta (pequeno para a base de 20 deixar espaço a outras rodadas)
MIN_RELEVANCE = 0.3    # corte da nota de relevância do JEV
MAX_CHUNKS = 15        # teto de contexto, não serve para ranquear
MAX_ROUNDS = 3         # rodadas de busca antes de desistir
MIN_SUFFICIENCY = 0.5  # nota de suficiência que encerra a busca
MAX_NEW_QUERIES = 3    # consultas que o LLM pode propor por rodada extra
```

> Numa base grande, o `TOP_K` por consulta deve subir para algo entre 10 e 30. O valor 5 existe só porque, com 20 trechos, um `TOP_K` maior faria a primeira rodada ver a base inteira e as rodadas seguintes não teriam nada novo para achar.

## Resultados: o filtro de relevância

Estes resultados são da primeira versão, com uma única rodada e `TOP_K = 20` (o JEV avaliando a base inteira). Eles isolam o efeito do filtro de relevância.

A base tem de propósito vários distratores: trechos do mesmo assunto que não respondem à pergunta (5 sobre o Cristo, 2 sobre o Pão de Açúcar, 2 sobre Brasília etc.).

**"Quando o Cristo Redentor foi inaugurado e qual a sua altura?"**

| Trecho | Similaridade | JEV | Mantido |
|---|---|---|---|
| cristo-1 (inauguração em 1931) | 0.748 | 0.76 | ✅ |
| cristo-2 (30 m de altura) | 0.709 | 0.76 | ✅ |
| cristo-3 (quem projetou) | 0.605 | 0.10 | ❌ |
| cristo-5 (raios) | 0.549 | 0.03 | ❌ |
| cristo-4 (Sete Maravilhas) | 0.538 | 0.08 | ❌ |

Resposta: *"inaugurado em 12 de outubro de 1931 e tem 30 metros de altura, sem contar o pedestal de 8 metros [cristo-1] [cristo-2]"*

**"Quem projetou os prédios de Brasília?"**

| Trecho | Similaridade | JEV | Mantido |
|---|---|---|---|
| brasilia-1 (inauguração, urbanismo de Lúcio Costa) | 0.665 | 0.52 | ✅ |
| brasilia-2 (Niemeyer projetou os edifícios) | 0.661 | 0.99 | ✅ |
| cristo-3 (projetista do Cristo) | 0.534 | 0.03 | ❌ |
| eiffel-2 (Gustave Eiffel) | 0.458 | 0.02 | ❌ |

Resposta: *"Oscar Niemeyer projetou os principais edifícios de Brasília… [brasilia-2]"*

**"Qual a população de Manaus?"** (a base não tem essa informação)

| Trecho | Similaridade | JEV | Mantido |
|---|---|---|---|
| amazonia-2 (rio Amazonas) | 0.385 | 0.01 | ❌ |
| amazonia-4 (Teatro Amazonas, em Manaus) | 0.373 | 0.03 | ❌ |
| amazonia-1 (área da floresta) | 0.351 | 0.02 | ❌ |

Resposta: *"Nenhum trecho relevante encontrado para responder a pergunta."*

**Resumo:** o JEV manteve 2, 2 e 0 dos 20 trechos avaliados. Os relevantes tiveram notas entre 0.52 e 0.99, e os demais ficaram em 0.10 ou menos.

## Resultados: a busca iterativa

Versão atual: `TOP_K = 5` por consulta, até 3 rodadas.

**Pergunta de três partes em que a primeira busca não basta:** *"Qual a altura do Cristo Redentor, quem trouxe o café ao Brasil e qual a vazão do rio Amazonas?"*

Rodada 1, buscando a pergunta original:

| Trecho | Similaridade | JEV | Mantido |
|---|---|---|---|
| cafe-3 (Melo Palheta trouxe o café) | 0.451 | 0.92 | ✅ |
| cristo-2 (30 m de altura) | 0.576 | 0.91 | ✅ |
| cristo-1 (inauguração) | 0.566 | 0.19 | ❌ |
| cristo-3 (quem projetou) | 0.486 | 0.19 | ❌ |
| cristo-5 (raios) | 0.475 | 0.05 | ❌ |

**Suficiência: 0.03, não basta.** Nenhum trecho sobre o rio Amazonas apareceu: o Cristo dominou os 5 vizinhos mais próximos.

Rodada 2: o LLM gerou três consultas voltadas só à parte que faltava:

- *Vazão média do rio Amazonas em metros cúbicos por segundo*
- *Quantidade de água que o rio Amazonas despeja no oceano*
- *Volume de fluxo do rio Amazonas durante o ano*

Elas trouxeram 15 candidatos novos. O JEV manteve só o `amazonia-2` (similaridade 0.663, JEV 0.94) e rejeitou os outros 14 com notas de 0.08 ou menos. **Suficiência: 0.95, basta.**

Resposta: *"A altura do Cristo Redentor é de 30 metros, sem contar o pedestal de 8 metros [cristo-2]. O café foi trazido ao Brasil por Francisco de Melo Palheta, procedente da Guiana Francesa [cafe-3]. A vazão do rio Amazonas é de cerca de 209 mil metros cúbicos por segundo [amazonia-2]."*

Com a busca vetorial pura, a terceira parte teria ficado sem resposta.

**Outros casos testados:**

| Pergunta | Rodadas | Suficiência final | Trechos usados |
|---|---|---|---|
| Inauguração e altura do Cristo | 1 | 0.96 | cristo-1, cristo-2 |
| Inauguração do Cristo + quem projetou os prédios de Brasília | 1 | 0.96 | brasilia-1, brasilia-2, cristo-1, cristo-3 |
| Inauguração do Cristo + prédios de Brasília + bondinho do Pão de Açúcar | 1 | 0.84 | pao-1, brasilia-1, brasilia-2, cristo-1, cristo-3 |
| Altura do Cristo + ano em que o café chegou | 1 | 0.94 | cafe-3, cristo-1, cristo-2 |
| **Altura do Cristo + quem trouxe o café + vazão do Amazonas** | **2** | **0.95** | amazonia-2, cafe-3, cristo-2 |
| População de Manaus (não está na base) | 2 (base esgotada) | 0.00 | nenhum. Resposta: "não sei" |

- Quando a primeira rodada basta, o laço não gasta nada a mais: é só uma chamada extra de suficiência.
- No caso de Manaus, a rodada 2 reformulou a pergunta de três jeitos ("número de habitantes", "dados demográficos", "população atual"). Todos os trechos restantes receberam 0.02 ou menos, a base se esgotou e o sistema respondeu que não sabe, sem inventar.

## Vantagens em relação à busca vetorial pura

A ideia central: **a busca vetorial não garante que um trecho responda à pergunta.** Ela mede apenas semelhança semântica, ou seja, se o trecho fala de algo parecido. O JEV acrescenta a peça que falta: uma nota absoluta de *"isto ajuda a responder"*. Todas as vantagens abaixo vêm daí.

1. **Separa "mesmo assunto" de "responde à pergunta".** No exemplo do Cristo, os três distratores tinham similaridade entre 0.54 e 0.61, bem acima do resto da base. Qualquer top-5 vetorial os teria mandado ao LLM. O JEV deu a eles notas de 0.03 a 0.10.

2. **Deixa de usar um top-K fixo.** Na busca pura, é preciso escolher K sem saber quantos trechos são úteis. Com K pequeno, a informação pode ficar de fora. Com K grande, entra ruído. Aqui a quantidade se ajusta à pergunta: 2, 2 e 0 trechos nos testes.

3. **Sabe dizer "não sei".** A busca vetorial sempre devolve os K vizinhos mais próximos, mesmo quando nenhum serve. Na pergunta sobre Manaus, o trecho do Teatro Amazonas cita Manaus e ficou no topo. Mandado ao LLM, abriria espaço para uma resposta inventada. Com o JEV, nenhum trecho passa e o sistema responde que não encontrou a informação.

4. **Corrige a ordem quando a similaridade empata.** Em Brasília, a similaridade praticamente empatou (0.665 contra 0.661) e colocou em primeiro o trecho menos útil. O JEV separou com clareza: 0.99 para o trecho do Niemeyer e 0.52 para o outro.

5. **Contexto menor e mais limpo para o LLM.** Com menos trechos, o prompt fica mais barato, a chance de o modelo se confundir com informação parecida mas errada é menor, e as citações ficam mais precisas.

6. **O corte usa uma escala que se entende.** A nota do JEV é uma probabilidade de relevância (0 a 1) que varia pouco entre perguntas. Já a similaridade de cosseno muda de escala conforme o modelo de embeddings e a pergunta: os relevantes ficaram em torno de 0.66 a 0.75, e os irrelevantes chegaram a 0.61. Por isso não existe um bom limiar fixo de similaridade.

7. **Dá um critério de parada, o que permite continuar buscando antes de dizer "não sei".** A busca vetorial pura não sabe quando parar: devolve os K vizinhos e acabou, sirvam eles ou não. Com a nota de suficiência do JEV, o sistema sabe se já tem o necessário. Se não tem, reformula e busca de novo, e só depois de esgotar as rodadas responde que não sabe. Na pergunta de três partes, isso recuperou a vazão do Amazonas, que a primeira busca não trouxe (veja [Resultados: a busca iterativa](#resultados-a-busca-iterativa)).

8. **Responde perguntas compostas por completo.** A suficiência é avaliada sobre o conjunto dos trechos e sobre *todas as partes* da pergunta. Por isso, achar duas de três respostas não encerra a busca. A nova rodada procura só o que falta.

9. **Não pesa na latência.** Com 20 chamadas em paralelo, o JEV avaliou todos os candidatos em 0.73 s, praticamente o mesmo tempo de gerar um embedding da pergunta (0.65 s) e menos de um terço do tempo do LLM (2.52 s). Veja [Custo e latência medidos](#custo-e-latência-medidos).

## Custo e latência medidos

Medidos numa única execução com a pergunta do Cristo Redentor, avaliando 20 candidatos. São valores de rede, então variam de uma execução para outra.

| Etapa | Tempo | Custo |
|---|---|---|
| JEV, 1 chamada (sequencial) | 0.45 a 0.69 s | ~US$ 0,000014 |
| JEV, 20 chamadas em paralelo (10 de cada vez) | 1.07 s | US$ 0,00028 |
| **JEV, 20 chamadas em paralelo (20 de cada vez)** | **0.73 s** | US$ 0,00028 |
| Embedding da pergunta | 0.65 s | desprezível |
| LLM, resposta final (`gpt-4.1-mini`) | 2.52 s | US$ 0,00013 |

- **Latência:** não é uma desvantagem real. Como as chamadas são independentes, avaliar 20 candidatos leva o tempo de uma chamada, e não de vinte. Aumentar o paralelismo de 10 para 20 reduziu o tempo de 1.07 s para 0.73 s, e o `rag.py` usa até 20 chamadas simultâneas.
- **Custo:** baixo em números absolutos, cerca de **US$ 0,28 a cada mil perguntas** avaliando 20 candidatos. Neste teste, porém, o JEV custou cerca de 2 vezes o LLM, porque o contexto enviado ao LLM era mínimo (2 trechos curtos). Com trechos reais e maiores, o LLM fica mais caro e o JEV passa a economizar, já que reduz o contexto que chega a ele. É preciso medir num cenário real antes de afirmar que o JEV é "muito mais barato que o LLM".
- **Crescimento:** é uma chamada por candidato. O custo cresce com o `TOP_K` e com o número de rodadas, não com o tamanho da base. A busca vetorial continua necessária para reduzir milhares de trechos a algumas dezenas.
- **Custo de uma rodada extra (estimativa, não medido):** uma chamada ao LLM para gerar consultas, de 1 a 3 embeddings, até `3 × TOP_K` chamadas ao JEV em paralelo e mais uma de suficiência. Somando os valores acima, algo entre 3 e 4 s, dominado pela chamada ao LLM. Quando a primeira rodada basta, o único acréscimo é a chamada de suficiência (~0.5 s).

## Limitações

- **O JEV só avalia o que a busca trouxe.** A busca iterativa atenua isso, porque consultas reformuladas trazem candidatos novos. Mas, se nenhuma consulta alcançar o trecho certo em `MAX_ROUNDS` rodadas, ele nunca é avaliado.
- **As novas consultas dependem do LLM.** Se ele reformular mal, a rodada extra não traz nada útil. O teste de suficiência evita respostas erradas nesse caso, mas não recupera o trecho.
- **Casos no limite.** O `brasilia-1` ficou entre 0.52 e 0.55 em execuções diferentes, e o `cristo-3` ("quem projetou o Cristo") passou com 0.31 a 0.34 em perguntas que falavam de quem projetou os prédios de Brasília. O corte em 0.3 prefere mandar um trecho a mais ao LLM do que perder um útil. O LLM ignorou esses trechos nas respostas.
- **Suficiência é uma estimativa.** Se o JEV disser "basta" cedo demais, a busca para antes da hora. Se nunca disser, gasta todas as rodadas. Nos testes, ela foi alta (0.84 a 0.96) quando havia tudo e baixa (0.00 a 0.03) quando faltava algo.
- **Base de teste pequena.** Com 20 trechos, a segunda rodada esgota a base rapidamente. O comportamento com milhares de trechos ainda precisa ser medido.

## Experimento: JEV como portão

A tese: um RAG que olha só o top-1 costuma acertar. O JEV entra como camada intermediária que confere se o trecho do top-1 basta. No caso comum ele aprova, e o custo é uma chamada ao JEV. Nos casos raros ele barra e examina os próximos vizinhos, um de cada vez, até a suficiência passar.

`RAG.ask_gate()` faz uma chamada ao JEV por trecho, com duas perguntas no mesmo estado (relevância do trecho novo e suficiência do conjunto). A busca para quando a suficiência passa do limiar, ao chegar a 5 trechos, ou depois de 3 trechos seguidos com relevância perto de zero (pergunta provavelmente sem resposta).

**Sistemas comparados** (`gate_experiment.py`):

| Sistema | Papel |
|---|---|
| top1 | baseline: um trecho e resposta direta |
| jev_gate | o portão proposto |
| sim_gate | controle principal: o mesmo portão decidindo pela similaridade de cosseno. Se empatar com o JEV, o JEV não é necessário. |
| topk_eq | controle de orçamento: top-k com k igual à média de trechos que o JEV examina. Mostra se o ganho é só "olhar mais". |
| oracle | teto: os trechos gold como contexto |

**Métricas:** matriz de confusão da decisão "o top-1 basta?", respostas corretas segundo o juiz, alucinação, tokens de contexto, latência p50/p95, percentual no caminho rápido, teste de McNemar e IC 95% por bootstrap contra o top1 e contra os controles, e comparação entre o caso comum e o caso raro.

**Fases:**

1. **Ajuste** num conjunto separado de perguntas: os limiares do sim_gate (só similaridade), o limiar de suficiência do JEV (uma chamada por pergunta) e o k do topk_eq. O resultado fica salvo em `results/<prefixo>_tuning.json` e é reaproveitado nas execuções seguintes.
2. **Avaliação:** os sistemas rodam em sequência para cada pergunta, para a latência não se misturar, e o juiz avalia em paralelo. Cada pergunta concluída vai para `results/<prefixo>_partial.jsonl`. Se a execução cair, basta rodar o mesmo comando para continuar de onde parou.
3. **Relatório:** `results/<prefixo>_report.md` e `results/<prefixo>_results.json`.

**Primeira execução, na base pequena (16 perguntas):** 94% de respostas corretas contra 56% do top1 (McNemar p = 0.031). O portão nunca aprovou um top-1 ruim, e o custo no caminho rápido foi +0.5 s. Mas o top-2 teve a mesma acurácia, porque quase todos os casos difíceis eram perguntas de duas partes. Relatório: [results/gate_report.md](results/gate_report.md).

### Rodar no SQuAD 2.0

```bash
python prepare_squad.py      # baixa o dev set e gera data/squad/ (sem chamar APIs; já executado)

# estimativa de chamadas, tempo e custo, sem chamar APIs
python gate_experiment.py --data data/squad/corpus.jsonl --lang en --cache \
    --questions data/squad/questions_test.json \
    --tune-questions data/squad/questions_tune.json --prefix squad --dry-run

# teste rápido com 10 perguntas (roda o ajuste completo e guarda o resultado)
python gate_experiment.py --data data/squad/corpus.jsonl --lang en --cache \
    --questions data/squad/questions_test.json \
    --tune-questions data/squad/questions_tune.json --prefix squad_smoke --limit 10

# experimento completo: 300 perguntas, ~80 min, ~US$ 5 (estimativa)
python gate_experiment.py --data data/squad/corpus.jsonl --lang en --cache \
    --questions data/squad/questions_test.json \
    --tune-questions data/squad/questions_tune.json --prefix squad
```

- **Base:** 1.204 parágrafos do SQuAD 2.0 dev, com o título do artigo antes de cada parágrafo.
- **Perguntas:** 100 de ajuste e 300 de teste, com 20% sem resposta. A proporção muda com `--unanswerable-frac` no `prepare_squad.py`. O SQuAD tem cerca de 50%, e tráfego real costuma ter menos.
- **Verdade de "o top-1 basta":** o trecho é o gold ou contém uma das respostas aceitas, porque no SQuAD o mesmo fato aparece em mais de um parágrafo do mesmo artigo.
- **Idioma:** com `--lang en`, os prompts do JEV e do gerador ficam em inglês. O juiz continua com o prompt em português, e as respostas de referência ficam em inglês.
- **Cache:** com `--cache`, os embeddings do corpus ficam salvos em `data/squad/corpus.<hash>.npy`, e a base não é embutida de novo a cada execução.
- **Limite de requisições:** o juiz (`claude-sonnet-5`) está limitado a 20 requisições por minuto para contas novas no OpenRouter, e é isso que define a duração.

## Como rodar

Requisitos: Python 3.10+, `faiss-cpu` e `numpy`, e uma chave do OpenRouter no `.env`:

```text
apikey=sk-or-...
```

```bash
python test_jev.py                                # testa noul, choice e score
python rag.py                                     # pergunta padrão (Cristo Redentor)
python rag.py "Qual a altura do Cristo Redentor, quem trouxe o café ao Brasil e qual a vazão do rio Amazonas?"
```

A saída mostra, para cada rodada, as consultas feitas, a nota de relevância e de similaridade de cada candidato, e a nota de suficiência.

No Windows, se os acentos saírem quebrados no terminal, defina `PYTHONIOENCODING=utf-8` antes de rodar.
