# 4. Discussão crítica

## 4.1 Limitações do melhor modelo

**1. Estimativa de desempenho otimista e sem incerteza.** O F1 de CV do modelo final (0,626) é o melhor entre 12 configurações e 2 pipelines, avaliados nos *mesmos* folds usados para escolher; esse máximo é enviesado para cima (Varma & Simon, 2006; Cawley & Talbot, 2010). Além disso, o resultado no teste é uma estimativa pontual: com 1.537 positivos, o intervalo de confiança do F1 e do recall não é desprezível. Seria preciso usar validação cruzada aninhada e IC por *bootstrap* para reportar o desempenho com incerteza.

**2. Uso de atributos sensíveis e risco de viés.** `sex`, `race` e `native.country` são features do modelo, e `marital.status` e `relationship` funcionam como proxies. O modelo reproduz desigualdades históricas: vimos que `sex_Female` tem coeficiente negativo grande (3.1) e contribui negativamente em predições individuais (3.4). Remover a variável sensível não basta, pois outras a reconstroem (Barocas, Hardt & Narayanan, 2019). Seria necessário medir desempenho por grupo (equal opportunity, paridade demográfica) antes de qualquer uso que afete pessoas.

## 4.2 Limitações dos dados

**1. Defasagem temporal e representatividade.** O Adult vem do censo americano de 1994 (Kohavi, 1996). Estrutura salarial, escolaridade e participação por gênero mudaram muito, e o corte de US\$ 50 mil não está corrigido pela inflação. O modelo não deve ser extrapolado para outros períodos ou países. Ding et al. (2021) criticam o uso do Adult como *benchmark* justamente por isso.

**2. Poucas variáveis, rótulo binarizado e problemas de qualidade.** A versão usada tem 11 features e não inclui `capital-gain`, `capital-loss` nem `hours-per-week` da versão original do UCI, o que limita o teto de desempenho (os erros do caso 3 sugerem isso). O alvo é a binarização de uma variável contínua, tornando difícil classificar quem está próximo do corte. Há ainda problemas de qualidade: o valor-lixo `"occupation"` na coluna `occupation` e `"?"` provavelmente não aleatórios (3.1 e 3.3), que o modelo aprende como se fossem informação.

## 4.3 Operacionalização e manutenção

**1. Monitoramento de *drift* e retreino.** Em produção, tanto a distribuição das entradas (*data drift*) quanto a relação entrada–renda (*concept drift*) mudam (Gama et al., 2014). Propõe-se monitorar a distribuição das features (ex.: PSI) e as métricas (F1 e recall de `>50K`) em dados rotulados recentes, com alertas e retreino agendado ou disparado por limiar, versionando dados, código e modelo para permitir reprodução e *rollback*; sem isso, o acúmulo de dívida técnica degrada o sistema silenciosamente (Sculley et al., 2015).

**2. Serviço versionado, limiar configurável e auditoria de equidade.** Empacotar o `Pipeline` completo (pré-processamento + modelo) em uma API, para aplicar em produção a mesma transformação do treino. O limiar de decisão deve ser parâmetro configurável, definido pelo custo relativo dos erros (hoje, FN = 662 contra FP = 378 com limiar 0,5). Registrar as predições, auditar periodicamente o desempenho por subgrupo (sexo, raça) e manter explicações SHAP para revisão humana de decisões contestadas.

## Conclusão

O melhor modelo foi a **Random Forest** (`n_estimators=200`, `min_samples_leaf=3`, `max_features=0.3`, sem limite de profundidade) com o **pipeline v2** (sem `fnlwgt`), com F1 de validação cruzada = 0,626. No teste, obteve **accuracy 0,837, precision 0,698, recall 0,569, F1 0,627 e ROC-AUC 0,878**, consistente com a validação, sem sinais de vazamento. O Dummy (F1 = 0) confirma que o problema exige aprendizado, e a LogReg é uma alternativa quase equivalente (F1 = 0,618) e mais interpretável.

Remover `fnlwgt` não afetou o desempenho dentro do ruído da validação, mas simplificou o pipeline e é coerente com a EDA, em que a variável é apenas um peso amostral. A curva de aprendizado mostra sobreajuste moderado (gap ≈ 10 p.p.) e saturação: mais dados do mesmo tipo ajudariam pouco. Escolaridade, estado civil (casado no civil / marido) e idade são os determinantes mais consistentes entre LogReg, importância de impureza e SHAP, em linha com a APS1. Os erros do modelo são principalmente falsos negativos (662 vs. 378 falsos positivos), e o limiar de 0,5 poderia ser ajustado conforme o custo dos erros.

As principais ressalvas são a estimativa de desempenho sem incerteza, o uso de atributos sensíveis (`sex` tem peso relevante), a defasagem dos dados de 1994, a ausência de variáveis fortes como `hours-per-week` e `capital-gain` e problemas de qualidade na coluna `occupation`. Em produção, seriam necessários monitoramento de drift, retreino e auditoria de equidade.
