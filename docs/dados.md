# 0. Dados e preparação (reaproveitando a APS1)

Carregamos o mesmo dataset e repetimos **exatamente** o tratamento da APS1:

- divisão treino/teste 80/20 com `stratify` e `random_state=42` (mesma divisão da APS1);
- `"?"` e `"occupation"` (valor-lixo na coluna `occupation`) viram `NaN`;
- `education` é descartada por ser redundante com `education.num`;
- o target `income` (texto) é convertido para **0/1**, sendo `1 = ">50K"` a classe positiva (minoritária).

O conjunto de **teste só será usado na Seção 2**. Toda escolha de modelo e hiperparâmetros usa apenas validação cruzada no treino.

??? example "Ver código"

    ```python
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score,
                                         GridSearchCV, RandomizedSearchCV, learning_curve)
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                                 RocCurveDisplay, roc_auc_score)
    import shap

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    RANDOM_STATE = 42
    ```


??? example "Ver código"

    ```python
    import kagglehub
    path = kagglehub.dataset_download("anaghakp/adult-income-census")
    df = pd.read_csv(os.path.join(path, "adult income1.csv"), encoding="latin-1")
    print(df.shape)
    df.head()
    ```


```text
(31947, 12)
```

|    | age   | workclass   | fnlwgt   | education   | education.num   | marital.status   | occupation   | relationship   | race   | sex    | native.country   | income   |
|:---|:------|:------------|:---------|:------------|:----------------|:-----------------|:-------------|:---------------|:-------|:-------|:-----------------|:---------|
| 0  | 17    | Private     | 148522   | 11th        | 7               | Never-married    | occupation   | Own-child      | White  | Male   | United-States    | <=50K    |
| 1  | 17    | Private     | 93235    | 12th        | 8               | Never-married    | occupation   | Own-child      | White  | Female | United-States    | <=50K    |
| 2  | 17    | Private     | 184924   | 9th         | 5               | Never-married    | occupation   | Own-child      | White  | Male   | United-States    | <=50K    |
| 3  | 17    | Private     | 116626   | 11th        | 7               | Never-married    | occupation   | Own-child      | White  | Male   | United-States    | <=50K    |
| 4  | 17    | Private     | 209949   | 11th        | 7               | Never-married    | occupation   | Own-child      | White  | Female | United-States    | <=50K    |


??? example "Ver código"

    ```python
    X_full = df.drop(columns=["income"])
    y_full = df["income"]

    # Mesma divisão da APS1
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full
    )

    # Limpeza (sem estado aprendido, logo não há vazamento): "?" / "occupation" -> NaN e remoção de education
    X_train_proc = X_train.drop(columns=["education"]).replace(["?", "occupation"], np.nan)
    X_test_proc  = X_test.drop(columns=["education"]).replace(["?", "occupation"], np.nan)

    # Target binário: 1 = ">50K" (classe positiva)
    print("Valores originais do target:", y_full.unique())
    y_tr = (y_train.astype(str).str.strip() == ">50K").astype(int)
    y_te = (y_test.astype(str).str.strip() == ">50K").astype(int)

    print(f"\nTreino: {X_train_proc.shape} | Teste: {X_test_proc.shape}")
    print("Proporção da classe >50K no treino:", round(y_tr.mean(), 3), "| no teste:", round(y_te.mean(), 3))
    ```


```text
Valores originais do target: <StringArray>
['<=50K', '>50K']
Length: 2, dtype: str

Treino: (25557, 10) | Teste: (6390, 10)
Proporção da classe >50K no treino: 0.24 | no teste: 0.241
```

## Escolha da métrica: **F1-score da classe `>50K`**

Na APS1 vimos que as classes são desbalanceadas (~76% `<=50K` e ~24% `>50K`). Com isso, a *accuracy* é enganosa: um modelo que sempre responde `<=50K` já acerta ~76% sem aprender nada. O F1 é a média harmônica entre precision e recall da classe positiva, então só é alto se o modelo realmente acerta os `>50K` sem gerar muitos falsos alarmes. Usaremos o **F1 como métrica única** em toda a Seção 1 (validação cruzada, busca de hiperparâmetros, comparação de pipelines e curvas de aprendizado), sempre com a **mesma validação**: `StratifiedKFold` com 5 folds.

??? example "Ver código"

    ```python
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    SCORING = "f1"
    ```
