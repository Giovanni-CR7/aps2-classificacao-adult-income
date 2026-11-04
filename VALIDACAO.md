# Validação do notebook APS2_Classificacao.ipynb

**Método:** li todas as células e conferi cada número citado no texto contra as saídas gravadas no notebook (tabelas, prints e os 9 gráficos). Recalculei métricas a partir da matriz de confusão e das tabelas por fold. **Não reexecutei o notebook** (depende de download do Kaggle); a validação é sobre as saídas gravadas.

## Veredito

Os números do texto batem com as saídas, a execução é sequencial (células 1–22) e sem erros, e a metodologia é correta (pré-processador dentro do `Pipeline`, mesma CV para tudo, teste usado uma vez). Há **5 pontos de redação/método que valem corrigir** antes de entregar e 4 menores.

## O que foi conferido e está correto

| Item | Conferência |
|---|---|
| Divisão e balanceamento | Treino 25.557 / teste 6.390; 24,0% e 24,1% de `>50K` |
| CV base | Dummy 0 · LogReg 0,6179 · RF 0,6052 (σ 0,0075 / 0,0057) |
| Busca de hiperparâmetros | 0,6184 (LogReg) e 0,6252 (RF); 10 combinações no grid, 12 de 225 (3×5×5×3) na busca aleatória |
| v1 vs v2 | Médias por fold recalculadas: 0,6252 e 0,6262; diferença +0,0010 |
| Teste | TN 4475 · FP 378 · FN 662 · TP 875 → precision 875/1253 = 0,698; recall 875/1537 = 0,569; accuracy 0,837; ROC-AUC 0,8779 |
| Coeficientes (3.1) | workclass_Missing −1,775; Own-child −1,266; Columbia −1,235; sex_Female −1,225 (4º) |
| Importância RF (3.2) | education.num 0,185; age 0,152; fnlwgt 0,138; Married-civ 0,136; Husband 0,103 |
| SHAP global (3.3) | 0,09 / 0,08 / 0,05 / 0,05 / 0,02 / 0,02 conforme o gráfico de barras |
| SHAP local (3.4) | Valores dos 3 waterfalls (f(x) = 0,624 / 0,008 / 0,698; base 0,24) conferem |
| Curva de aprendizado | Gap ≈ 10 p.p., treino 0,72–0,75, validação 0,615–0,628, plana: confere |
| Dados | 11 features (12 colunas − target); 89 colunas na v1 e 88 na v2 |

## Pontos a corrigir

### 1. LogReg vs RF: o resultado da comparação pareada não é reportado e a Conclusão o contradiz
A célula 21(b) imprime RF − LogReg = **+0,0067** (σ = 0,0054; RF melhor em 4/5 folds) e o rótulo "diferença maior que o ruído". O texto (1.2 e 1.3) promete essa verificação mas nunca informa o resultado, e a Conclusão chama a LogReg de "alternativa quase equivalente".
Um teste t pareado nos 5 folds (feito por mim) dá **t = 2,47, p ≈ 0,07**: inconclusivo. Os folds não são independentes e o critério `|média| < desvio` do código é heurístico. **Sugestão:** escrever "a RF teve F1 ~0,7 p.p. maior (melhor em 4 de 5 folds), diferença modesta e sem evidência forte com 5 folds" e ajustar a Conclusão.

### 2. A regra de escolha da v2 no código não é a justificativa do texto
O texto diz "adotamos a v2 por parcimônia (sem melhora, apenas sem perda)", mas o código escolhe por `busca_v2.best_score_ > v1` (0,6262 > 0,6252). O resultado é o mesmo, mas a regra é frágil: se a v2 desse 0,6249, o código escolheria a v1. Na comparação pareada, a v2 só foi ≥ em 1/5 folds (p ≈ 0,78 para a diferença). **Sugestão:** trocar para `>=` com tolerância (ex.: `v2 >= v1 - 0.002`) ou reescrever o texto dizendo que a escolha foi por critério de score.

### 3. Seção 2 exagera: "confirma que não houve overfitting nem vazamento"
F1 de teste ≈ F1 de CV é *compatível com* ausência de vazamento, mas não prova. A própria 4.1 diz que a estimativa de CV é otimista. **Sugestão:** "é compatível com ausência de overfitting de seleção e de vazamento".

### 4. 3.3: "o SHAP remove o artefato do `fnlwgt`" é impreciso
O SHAP foi calculado sobre a v2, que **não tem** `fnlwgt`. A mudança no ranking vem de a feature ter sido removida, não do SHAP. Para mostrar que o SHAP corrige o viés do MDI, seria preciso calcular SHAP também na v1.

### 5. Coluna `occupation`: o problema é diagnosticado, mas não tratado
A tabela da célula 32 mostra que `"?"` (n = 1.432; 10,2% `>50K`) e o valor-lixo `"occupation"` (n = 1.188; **53,9% `>50K`**) são grupos opostos, e o pipeline os junta em um único NaN → `Missing`. O modelo então aprende `occupation_Missing` com efeito **positivo** (+1,165 na LogReg e positivo no SHAP). O notebook reconhece isso (3.1, 3.3, 4.2), mas o "efeito" é provavelmente artefato do arquivo, não do mundo real. **Sugestão:** manter as duas categorias separadas (ex.: `"?"` → `Missing`, `"occupation"` → `Invalid`), refazer a v1/v2 e comparar. Se não houver tempo, deixar explícito na Conclusão que `occupation_Missing` não deve ser interpretada.

## Pontos menores

6. **Referências não citadas no texto:** Breiman (2001), Pedregosa et al. (2011) e Niculescu-Mizil & Caruana (2005) estão na lista, mas não aparecem no corpo. Cite-as (Breiman ao introduzir a RF; Pedregosa ao citar o scikit-learn; Niculescu-Mizil ao falar de probabilidades/limiar) ou remova.
7. **Célula 1 (`%pip install`) falhou** com "No module named pip" (o `shap` já estava instalado, então nada quebrou), mas o erro fica nas saídas. Troque por um `requirements.txt` ou rode com o pip correto.
8. **Caminho local nas saídas:** aparece `c:\Users\hecto\...` nas saídas do notebook. Limpe antes de publicar no GitHub. (Omiti essas saídas do site.)
9. **"1.778 registros `?` em `workclass`"** vem da APS1 e não dá para verificar neste notebook; confira se o número é da base completa ou do treino (no treino, `?` em `occupation` são 1.432).

## Limites desta validação
Não reexecutei o notebook, então não testei reprodutibilidade. Os números da APS1 (EDA) citados no texto não foram conferidos.
