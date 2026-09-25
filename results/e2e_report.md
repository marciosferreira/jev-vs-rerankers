# Experimento ponta a ponta: a política de contexto muda a alucinação?

Gerado em 2026-09-25 11:36. 299 perguntas de teste do SQuAD 2.0 (245 com resposta, 54 sem resposta), com os mesmos 10 candidatos por pergunta do experimento de rerank. Gerador: `openai/gpt-4.1-mini`. Juiz: `anthropic/claude-sonnet-5`. Uma execução.

1 pergunta(s) excluída(s) porque o filtro de segurança do juiz se recusou a avaliar alguma das respostas: 572a0e4b6aef051400155214.

- **top3:** os 3 trechos mais parecidos pelo embedding (prática comum).
- **cohere_top3:** os 3 melhores segundo o Cohere `rerank-v4.0-pro`.
- **cohere_cut:** todos os trechos com nota Cohere >= 0.891.
- **jev_cut:** todos os trechos com relevância JEV >= 0.97.

Os cortes foram escolhidos nas 100 perguntas de ajuste (maior F1), sem olhar as de teste. Quando uma política não mantém nenhum trecho, o gerador **não é chamado** e o sistema devolve uma mensagem fixa de "não encontrado".

Correta = o juiz considera todas as partes corretas e não há alucinação. Alucinação = o juiz marca que a resposta afirma algo fora do contexto recebido ou que contradiz a referência.

## 1. Resultado principal

| Métrica | top3 | cohere_top3 | cohere_cut | jev_cut |
|---|---|---|---|---|
| Respostas corretas (todas) | 81% | 82% | 79% | 84% |
| **Alucinação (todas)** | **8%** | **8%** | **2%** | **1%** |
| Corretas, perguntas com resposta (n=245) | 91% | 94% | 82% | 83% |
| Corretas, perguntas sem resposta (n=54) | 35% | 28% | 65% | 85% |
| **Alucinação nas perguntas sem resposta** (n=54) | **35%** | **30%** | **7%** | **4%** |
| "Não sei" indevido (perguntas com resposta) | 4% | 2% | 15% | 15% |
| Perguntas sem chamada ao gerador | 0% | 0% | 22% | 27% |

## 2. Comparação pareada com o jev_cut

| Contra | Métrica | jev_cut melhor | outro melhor | McNemar p | Diferença jev_cut − outro (IC 95%) |
|---|---|---|---|---|---|
| top3 | corretas | 33 | 26 | 0.435 | +2.3% (-2.7% a +7.4%) |
| top3 | alucinação | 22 | 1 | 0.000 | -7.0% (-10.0% a -4.0%) |
| cohere_top3 | corretas | 34 | 29 | 0.615 | +1.7% (-3.3% a +7.0%) |
| cohere_top3 | alucinação | 21 | 1 | 0.000 | -6.7% (-10.0% a -4.0%) |
| cohere_cut | corretas | 38 | 25 | 0.130 | +4.3% (-0.7% a +9.7%) |
| cohere_cut | alucinação | 6 | 2 | 0.289 | -1.3% (-3.3% a +0.3%) |

## 3. Custo e contexto

| Métrica | top3 | cohere_top3 | cohere_cut | jev_cut |
|---|---|---|---|---|
| Trechos enviados ao gerador (média) | 3.00 | 3.00 | 0.95 | 0.85 |
| Tokens de entrada do gerador (média, medido) | 605 | 602 | 222 | 200 |
| Gerador, US$ a cada mil perguntas (medido) | 0.341 | 0.341 | 0.150 | 0.133 |
| Seleção de contexto, US$ a cada mil perguntas | 0 | 2.500 | 2.500 | 0.256 |
| **Total, US$ a cada mil perguntas** | **0.341** | **2.841** | **2.650** | **0.389** |
| Latência do gerador p50 (s) | 2.10 | 2.11 | 1.89 | 1.89 |

Seleção de contexto: JEV medido (`usage.cost`, 11 chamadas por pergunta no experimento de rerank); Cohere a preço de tabela (US$ 2.50 a cada mil buscas). A latência do gerador inclui 0 s quando ele não é chamado.

## 4. Por que as políticas erram

Perguntas com resposta em que a política enviou ao menos um trecho útil (o gerador tinha como acertar):

| | top3 | cohere_top3 | cohere_cut | jev_cut |
|---|---|---|---|---|
| Com trecho útil no contexto | 92% | 96% | 84% | 84% |
| Corretas quando havia trecho útil | 98% | 97% | 97% | 99% |

## Limitações

- 299 perguntas de uma base (SQuAD 2.0), uma execução; 54 perguntas sem resposta.
- O juiz é um LLM e pode errar; a alucinação é a marcação dele.
- O custo da Cohere é preço de tabela de terceiros; o do JEV e o do gerador são medidos.
- Latência só do gerador: a seleção de contexto (JEV ~0,7 s, Cohere ~1 s) não está somada.
