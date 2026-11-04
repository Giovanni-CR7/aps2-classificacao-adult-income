# APS2: Classificação — Adult Income Census

**Membros:** Hector Mathias e Giovanni Rodrigues  
**Data da entrega:** 30/10/2026  
**Dataset:** [Adult Income Census (Kaggle)](https://www.kaggle.com/datasets/anaghakp/adult-income-census)  
**APS1 (EDA):** https://giovanni-cr7.github.io/aps1-eda-adult-income/  
**Repositório:** https://github.com/Giovanni-CR7/aps1-eda-adult-income

**Problema:** prever se a renda anual de uma pessoa é `>50K` ou `<=50K` (classificação binária) a partir de características demográficas e profissionais.

**Roteiro:** (1) seleção e implementação dos modelos → (2) avaliação no teste → (3) interpretação e explicabilidade → (4) discussão crítica → referências.

## Resultado em uma olhada

| Item | Valor |
|---|---|
| **Modelo final** | Random Forest (`n_estimators=200`, `min_samples_leaf=3`, `max_features=0.3`, `max_depth=None`) |
| **Pipeline** | v2 (sem `fnlwgt`) |
| **F1 (validação cruzada, 5 folds)** | 0,6262 |
| **F1 (teste, classe `>50K`)** | 0,627 |
| **Precision / Recall (`>50K`)** | 0,698 / 0,569 |
| **Accuracy** | 0,837 |
| **ROC-AUC** | 0,8779 |
| **Baseline (Dummy)** | F1 = 0 |

## Navegação

- [Dados e métrica](dados.md): divisão treino/teste, limpeza e por que usamos F1.
- [Modelos](modelos.md): Dummy, Regressão Logística e Random Forest; hiperparâmetros; pipeline v2; curva de aprendizado.
- [Avaliação no teste](avaliacao.md): métricas, matriz de confusão e ROC.
- [Interpretabilidade](interpretabilidade.md): coeficientes, importância de features e SHAP.
- [Discussão crítica](discussao.md): limitações, dados, operacionalização e conclusão.
- [Referências](referencias.md)

!!! note "Sobre o código"
    O código de cada etapa está em blocos recolhíveis (**Ver código**). As saídas e os gráficos são os do notebook original, executado em Python 3.14.
