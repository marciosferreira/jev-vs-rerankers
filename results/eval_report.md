# Avaliação: RAG top-5 x RAG com JEV

Gerado em 2026-09-24 15:52. Juiz: `anthropic/claude-sonnet-5`. Gerador (ambos): `openai/gpt-4.1-mini`. Embeddings (ambos): `openai/text-embedding-3-small`. 16 perguntas, uma execução.

- **baseline:** FAISS top-5 por similaridade, resposta direta, sem filtro.
- **jev:** filtro de relevância do JEV (>= 0.3), teste de suficiência (>= 0.5) e busca iterativa (até 3 rodadas, top-5 por consulta).

## Resultado geral

| Métrica | baseline | jev |
|---|---|---|
| Nota do juiz (1 a 5) | 4.88 | 5.00 |
| Completude (partes corretas) | 98% | 100% |
| Taxa de alucinação | 0% | 0% |
| Abstenção correta (perguntas sem resposta) | 3/3 | 3/3 |
| Recall dos trechos gold | 97% | 100% |
| Trechos no contexto (média) | 5.00 | 1.31 |
| Trechos não-gold no contexto (média) | 3.81 | 0.06 |
| Latência média (s) | 2.33 | 4.96 |
| Rodadas de busca (média) | 1.00 | 1.31 |

**Comparação direta (às cegas):** jev venceu 2, baseline venceu 0, empates 14.

## Por categoria

| Categoria | n | Nota baseline | Nota jev | Completude baseline | Completude jev | Vitórias jev / baseline / empate |
|---|---|---|---|---|---|---|
| simples | 6 | 5.00 | 5.00 | 100% | 100% | 0 / 0 / 6 |
| composta | 2 | 5.00 | 5.00 | 100% | 100% | 0 / 0 / 2 |
| multi-tema | 4 | 4.50 | 5.00 | 92% | 100% | 1 / 0 / 3 |
| sem resposta | 3 | 5.00 | 5.00 | 100% | 100% | 0 / 0 / 3 |
| parcial | 1 | 5.00 | 5.00 | 100% | 100% | 1 / 0 / 0 |

## Por pergunta

| # | Pergunta | Nota b | Nota j | Alucinou b / j | Recall gold b / j | Rodadas j | Vencedor |
|---|---|---|---|---|---|---|---|
| q01 | Quando o Cristo Redentor foi inaugurado? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q02 | Quem projetou os prédios de Brasília? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q03 | Qual a altura do morro do Pão de Açúcar? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q04 | Qual estado brasileiro mais produz café? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q05 | Quem esculpiu o rosto do Cristo Redentor? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q06 | Quais sistemas o INPE usa para monitorar o desmatamento da Amazônia? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q07 | Quando o Cristo Redentor foi inaugurado e qual a sua altura? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q08 | Quando foi inaugurado o bondinho do Pão de Açúcar e qual a altura do morro? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q09 | Qual a altura do Cristo Redentor, quem trouxe o café ao Brasil e qual a vazão do rio Amazonas? | 3 | 5 | não / não | 67% / 100% | 2 | jev |
| q10 | Em que ano o Maracanã foi inaugurado e em que ano Brasília foi inaugurada? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q11 | Que outra estrutura famosa Gustave Eiffel projetou e qual a altura da Torre Eiffel? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q12 | Qual a participação do Brasil na produção mundial de café e em quantos países está distribuída a Floresta Amazônica? | 5 | 5 | não / não | 100% / 100% | 1 | empate |
| q13 | Qual a população de Manaus? | 5 | 5 | não / não | - / - | 2 | empate |
| q14 | Quanto custa o ingresso para visitar o Cristo Redentor? | 5 | 5 | não / não | - / - | 2 | empate |
| q15 | Quem foi o arquiteto do Teatro Amazonas? | 5 | 5 | não / não | - / - | 2 | empate |
| q16 | Quando o Teatro Amazonas foi inaugurado e quantos lugares ele tem? | 5 | 5 | não / não | 100% / 100% | 2 | jev |

## Respostas e justificativas do juiz

### q01: Quando o Cristo Redentor foi inaugurado?

Referência: Em 12 de outubro de 1931.

**baseline** (nota 5, contexto: cristo-1, cristo-4, cristo-3, cristo-2, cristo-5)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1].

Juiz: A resposta indica corretamente a data de inauguração, conforme a referência, e cita a fonte adequadamente.

**jev** (nota 5, contexto: cristo-1)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 [cristo-1].

Juiz: A resposta indica corretamente a data de inauguração, 12 de outubro de 1931, conforme a referência e o contexto fornecido, sem inventar informações.

Comparação: **empate**. Ambas as respostas fornecem a data correta de inauguração com a mesma citação, sem diferenças relevantes.

### q02: Quem projetou os prédios de Brasília?

Referência: Oscar Niemeyer (Congresso Nacional, Palácio da Alvorada, Catedral Metropolitana).

**baseline** (nota 5, contexto: brasilia-1, brasilia-2, cristo-3, eiffel-2, cristo-1)

> Oscar Niemeyer projetou os principais edifícios de Brasília, como o Congresso Nacional, o Palácio da Alvorada e a Catedral Metropolitana [brasilia-2].

Juiz: A resposta identifica corretamente Oscar Niemeyer e cita os mesmos exemplos da referência, sem inventar informações.

**jev** (nota 5, contexto: brasilia-2, brasilia-1)

> Oscar Niemeyer projetou os principais edifícios de Brasília, como o Congresso Nacional, o Palácio da Alvorada e a Catedral Metropolitana [brasilia-2].

Juiz: A resposta identifica corretamente Oscar Niemeyer como projetista e cita os mesmos exemplos de edifícios da referência, sem inventar informações.

Comparação: **empate**. As duas respostas são idênticas e corretamente citam Oscar Niemeyer com os mesmos exemplos da resposta de referência.

### q03: Qual a altura do morro do Pão de Açúcar?

Referência: 396 metros.

**baseline** (nota 5, contexto: pao-2, cristo-1, cristo-2, pao-1, eiffel-1)

> A altura do morro do Pão de Açúcar é de 396 metros [pao-2].

Juiz: A resposta indica corretamente os 396 metros, conforme a referência, sem informações inventadas.

**jev** (nota 5, contexto: pao-2)

> A altura do morro do Pão de Açúcar é 396 metros [pao-2].

Juiz: A resposta indica corretamente os 396 metros de altura, conforme a referência e o contexto fornecido.

Comparação: **empate**. Ambas as respostas informam corretamente a altura de 396 metros com a mesma citação, diferindo apenas em fraseado sem alterar correção ou completude.

### q04: Qual estado brasileiro mais produz café?

Referência: Minas Gerais.

**baseline** (nota 5, contexto: cafe-2, cafe-1, cafe-3, cafe-4, amazonia-1)

> O estado brasileiro que mais produz café é Minas Gerais [cafe-2].

Juiz: A resposta identifica corretamente Minas Gerais como o estado que mais produz café, de acordo com a referência e o contexto fornecido.

**jev** (nota 5, contexto: cafe-2)

> O estado brasileiro que mais produz café é Minas Gerais [cafe-2].

Juiz: A resposta identifica corretamente Minas Gerais como o estado que mais produz café, alinhando-se com a referência e o contexto fornecido.

Comparação: **empate**. As duas respostas são idênticas e corretas, informando que Minas Gerais é o maior produtor de café.

### q05: Quem esculpiu o rosto do Cristo Redentor?

Referência: O escultor romeno Gheorghe Leonida.

**baseline** (nota 5, contexto: cristo-3, cristo-5, cristo-1, cristo-2, cristo-4)

> O rosto do Cristo Redentor foi esculpido pelo romeno Gheorghe Leonida [cristo-3].

Juiz: A resposta identifica corretamente o escultor Gheorghe Leonida, coincidindo com a referência e citando o contexto adequado.

**jev** (nota 5, contexto: cristo-3)

> O rosto do Cristo Redentor foi esculpido pelo romeno Gheorghe Leonida [cristo-3].

Juiz: A resposta identifica corretamente Gheorghe Leonida como o escultor do rosto, conforme a referência e o contexto fornecido.

Comparação: **empate**. As duas respostas são idênticas e correspondem exatamente à resposta de referência.

### q06: Quais sistemas o INPE usa para monitorar o desmatamento da Amazônia?

Referência: PRODES e DETER, por satélite.

**baseline** (nota 5, contexto: amazonia-3, amazonia-1, amazonia-2, brasilia-2, amazonia-4)

> O INPE usa os sistemas PRODES e DETER para monitorar o desmatamento na Amazônia brasileira [amazonia-3].

Juiz: A resposta identifica corretamente os sistemas PRODES e DETER, citando a fonte adequada, correspondendo integralmente à resposta de referência.

**jev** (nota 5, contexto: amazonia-3)

> O INPE usa os sistemas PRODES e DETER para monitorar o desmatamento na Amazônia brasileira, através de satélites [amazonia-3].

Juiz: A resposta identifica corretamente os sistemas PRODES e DETER usados pelo INPE via satélite, alinhando-se completamente com a resposta de referência e o contexto fornecido.

Comparação: **empate**. Ambas respondem corretamente citando PRODES e DETER; a menção ao satélite em B é um detalhe extra que não torna a resposta melhor.

### q07: Quando o Cristo Redentor foi inaugurado e qual a sua altura?

Referência: Inaugurado em 12 de outubro de 1931; tem 30 metros de altura (sem o pedestal de 8 metros).

**baseline** (nota 5, contexto: cristo-1, cristo-2, cristo-3, cristo-5, cristo-4)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 e tem 30 metros de altura, sem contar o pedestal de 8 metros [cristo-1] [cristo-2].

Juiz: A resposta informa corretamente a data de inauguração e a altura da estátua, conforme a referência e o contexto fornecido, sem inventar informações.

**jev** (nota 5, contexto: cristo-2, cristo-1)

> O Cristo Redentor foi inaugurado em 12 de outubro de 1931 e tem 30 metros de altura, sem contar o pedestal de 8 metros [cristo-1] [cristo-2].

Juiz: A resposta contém corretamente a data de inauguração e a altura da estátua, ambas coerentes com o contexto e a referência.

Comparação: **empate**. As duas respostas são idênticas e correspondem exatamente à resposta de referência.

### q08: Quando foi inaugurado o bondinho do Pão de Açúcar e qual a altura do morro?

Referência: O bondinho foi inaugurado em 1912; o morro tem 396 metros.

**baseline** (nota 5, contexto: pao-1, pao-2, cristo-1, cafe-3, brasilia-1)

> O bondinho do Pão de Açúcar foi inaugurado em 1912 e o morro do Pão de Açúcar tem 396 metros de altura [pao-1] [pao-2].

Juiz: A resposta contém corretamente a data de inauguração (1912) e a altura do morro (396 metros), ambas confirmadas pelo contexto e pela referência.

**jev** (nota 5, contexto: pao-1, pao-2)

> O bondinho do Pão de Açúcar foi inaugurado em 1912 e o morro do Pão de Açúcar tem 396 metros de altura [pao-1][pao-2].

Juiz: A resposta indica corretamente o ano de inauguração (1912) e a altura do morro (396 metros), ambos consistentes com o contexto e a referência.

Comparação: **empate**. Ambas as respostas fornecem as mesmas informações corretas e completas, diferindo apenas na formatação das citações.

### q09: Qual a altura do Cristo Redentor, quem trouxe o café ao Brasil e qual a vazão do rio Amazonas?

Referência: O Cristo tem 30 metros (sem o pedestal de 8 m); o café foi trazido por Francisco de Melo Palheta, em 1727, da Guiana Francesa; o Amazonas despeja cerca de 209 mil metros cúbicos por segundo no Atlântico.

**baseline** (nota 3, contexto: cristo-2, cristo-1, cristo-3, cristo-5, cafe-3)

> A altura da estátua do Cristo Redentor é de 30 metros, sem contar o pedestal de 8 metros [cristo-2]. O café chegou ao Brasil em 1727, trazido da Guiana Francesa por Francisco de Melo Palheta [cafe-3]. O contexto fornecido não informa a vazão do rio Amazonas.

Juiz: As partes sobre a altura do Cristo e a origem do café foram corretas, mas a resposta abstém-se sobre a vazão do Amazonas, informação que a referência indica estar disponível na base, configurando falha de recuperação.

**jev** (nota 5, contexto: amazonia-2, cafe-3, cristo-2)

> A altura do Cristo Redentor é de 30 metros, sem contar o pedestal, que tem 8 metros [cristo-2]. O café foi trazido ao Brasil em 1727 por Francisco de Melo Palheta, vindo da Guiana Francesa [cafe-3]. A vazão do rio Amazonas é de aproximadamente 209 mil metros cúbicos por segundo [amazonia-2].

Juiz: As três partes foram respondidas corretamente e de forma completa, condizentes com a resposta de referência e sem informações inventadas.

Comparação: **jev**. A responde corretamente às três perguntas com a informação da base, enquanto B omite a vazão do Amazonas que estava disponível na resposta de referência.

### q10: Em que ano o Maracanã foi inaugurado e em que ano Brasília foi inaugurada?

Referência: O Maracanã em 1950; Brasília em 1960 (21 de abril).

**baseline** (nota 5, contexto: maracana-1, brasilia-1, cristo-1, brasilia-2, amazonia-4)

> O Estádio do Maracanã foi inaugurado em 1950, e Brasília foi inaugurada em 1960 [maracana-1] [brasilia-1].

Juiz: A resposta indica corretamente os anos de inauguração do Maracanã (1950) e de Brasília (1960), coincidindo com a referência, embora não mencione o dia exato de Brasília, o que não é essencial.

**jev** (nota 5, contexto: brasilia-1, maracana-1)

> O Maracanã foi inaugurado em 1950 e Brasília foi inaugurada em 1960 [brasilia-1] [maracana-1].

Juiz: A resposta indica corretamente os anos de 1950 para o Maracanã e 1960 para Brasília, coincidindo com a referência e sem alucinações.

Comparação: **empate**. Ambas as respostas informam corretamente os anos de inauguração do Maracanã (1950) e de Brasília (1960), diferindo apenas na ordem das citações.

### q11: Que outra estrutura famosa Gustave Eiffel projetou e qual a altura da Torre Eiffel?

Referência: A estrutura interna da Estátua da Liberdade; a Torre Eiffel tem cerca de 330 metros.

**baseline** (nota 5, contexto: eiffel-2, eiffel-1, cristo-3, brasilia-2, cristo-2)

> Gustave Eiffel projetou a estrutura interna da Estátua da Liberdade, em Nova York. A Torre Eiffel tem cerca de 330 metros de altura. [eiffel-2] [eiffel-1]

Juiz: A resposta identifica corretamente a Estátua da Liberdade como outra estrutura projetada por Eiffel e informa a altura correta da Torre Eiffel, ambas fundamentadas no contexto fornecido.

**jev** (nota 5, contexto: eiffel-1, eiffel-2)

> Gustave Eiffel projetou a estrutura interna da Estátua da Liberdade, em Nova York. A Torre Eiffel tem cerca de 330 metros de altura [eiffel-1][eiffel-2].

Juiz: A resposta identifica corretamente a Estátua da Liberdade e a altura de 330 metros, ambas as informações presentes no contexto e alinhadas com a referência.

Comparação: **empate**. Ambas as respostas são igualmente corretas, completas e apresentam as mesmas informações, diferindo apenas na ordem das citações.

### q12: Qual a participação do Brasil na produção mundial de café e em quantos países está distribuída a Floresta Amazônica?

Referência: Cerca de um terço da produção global de café; a Floresta Amazônica está distribuída entre nove países.

**baseline** (nota 5, contexto: cafe-1, amazonia-1, cafe-2, cafe-3, amazonia-3)

> O Brasil é responsável por cerca de um terço da produção mundial de café [cafe-1]. A Floresta Amazônica está distribuída entre nove países [amazonia-1].

Juiz: A resposta cobre corretamente ambas as partes: um terço da produção mundial de café e nove países para a Floresta Amazônica, ambas presentes no contexto e alinhadas com a referência.

**jev** (nota 5, contexto: cafe-1, amazonia-1)

> O Brasil é responsável por cerca de um terço da produção global de café, e a Floresta Amazônica está distribuída entre nove países [cafe-1] [amazonia-1].

Juiz: A resposta cobre corretamente ambas as partes: um terço da produção mundial de café e nove países para a Amazônia, conforme a referência e o contexto.

Comparação: **empate**. Ambas as respostas fornecem corretamente as mesmas informações (um terço da produção mundial de café e nove países da Amazônia) com as citações adequadas.

### q13: Qual a população de Manaus?

Referência: A base não contém essa informação. A resposta correta é dizer que não sabe.

**baseline** (nota 5, contexto: amazonia-2, amazonia-4, amazonia-1, amazonia-3, maracana-1)

> O contexto fornecido não informa a população de Manaus, portanto, não sei essa parte.

Juiz: A base não contém a população de Manaus, e o sistema corretamente respondeu que não sabe, sem inventar informações.

**jev** (nota 5, contexto: vazio)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A referência indica que a base não contém a informação, e o sistema corretamente afirma não ter encontrado dados relevantes, sem alucinar.

Comparação: **empate**. Ambas as respostas corretamente indicam que a informação não está disponível na base, sem inventar dados.

### q14: Quanto custa o ingresso para visitar o Cristo Redentor?

Referência: A base não contém essa informação. A resposta correta é dizer que não sabe.

**baseline** (nota 5, contexto: cristo-1, cristo-2, cristo-5, cristo-4, cristo-3)

> O contexto fornecido não informa o valor do ingresso para visitar o Cristo Redentor, portanto, não sei essa parte.

Juiz: A resposta corretamente indica que a informação sobre o preço do ingresso não está disponível na base, alinhando-se com a referência.

**jev** (nota 5, contexto: vazio)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta corretamente indica que não há informação disponível, alinhando-se com a referência que afirma que a base não contém esse dado.

Comparação: **empate**. Ambas as respostas indicam corretamente que a base não contém a informação solicitada, sem inventar dados.

### q15: Quem foi o arquiteto do Teatro Amazonas?

Referência: A base não contém essa informação (ela só diz que o teatro foi inaugurado em 1896, no ciclo da borracha). A resposta correta é dizer que não sabe.

**baseline** (nota 5, contexto: amazonia-4, cristo-3, brasilia-2, eiffel-2, brasilia-1)

> O contexto fornecido não informa quem foi o arquiteto do Teatro Amazonas, apenas menciona que ele foi inaugurado em 1896 durante o ciclo da borracha, com materiais importados da Europa [amazonia-4]. Portanto, não sei essa parte.

Juiz: A resposta corretamente afirma não saber quem foi o arquiteto, alinhando-se com a referência, sem inventar informações.

**jev** (nota 5, contexto: vazio)

> Nenhum trecho relevante encontrado para responder a pergunta.

Juiz: A resposta corretamente indica que não encontrou a informação, alinhando-se com a referência de que a base não contém o nome do arquiteto.

Comparação: **empate**. Ambas as respostas corretamente indicam que não sabem quem foi o arquiteto, sem inventar informação, apenas com formulações diferentes.

### q16: Quando o Teatro Amazonas foi inaugurado e quantos lugares ele tem?

Referência: Foi inaugurado em 1896. A base não informa a capacidade; a resposta correta diz que não sabe essa parte.

**baseline** (nota 5, contexto: amazonia-4, maracana-1, cristo-1, brasilia-1, amazonia-2)

> O Teatro Amazonas foi inaugurado em 1896, durante o ciclo da borracha. No entanto, o contexto fornecido não informa quantos lugares ele tem [amazonia-4].

Juiz: A resposta acerta a data de inauguração e corretamente admite não saber a capacidade, que de fato não está na base conforme a referência.

**jev** (nota 5, contexto: amazonia-4)

> O Teatro Amazonas foi inaugurado em 1896 [amazonia-4]. O contexto não informa quantos lugares ele tem.

Juiz: A resposta informa corretamente a data de inauguração e abstém-se corretamente sobre a capacidade, que não consta na base, alinhando-se à referência.

Comparação: **jev**. B responde apenas com o que a base permite, enquanto A adiciona um detalhe ('ciclo da borracha') não verificado pela citação e não solicitado.

## Limitações desta avaliação

- 16 perguntas e uma única execução: diferenças pequenas podem ser ruído.
- Base de conhecimento de 20 trechos, escrita para este teste.
- O juiz é um LLM e pode errar. A comparação direta usa ordem aleatória para reduzir viés de posição.
- Recall e trechos não-gold são objetivos (comparados com os ids gold de cada pergunta) e não dependem do juiz.
- Uma primeira execução foi descartada por falhas da própria avaliação: o juiz aceitava "não sei" quando o contexto recuperado não tinha a informação (mesmo existindo na base), a comparação direta premiava detalhes não perguntados, e a referência da q02 exigia algo que a pergunta não pedia. Os prompts do juiz e a referência foram corrigidos; o sistema RAG não foi alterado.
