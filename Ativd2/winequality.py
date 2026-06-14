import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

red   = pd.read_csv("winequality-red.csv",   sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

red["wine_type"]   = "red"
white["wine_type"] = "white"

df = pd.concat([red, white], ignore_index=True)
df.to_csv("winequality_combined.csv", index=False)

print("Formato da base unificada:", df.shape)
print(df.head())
print("\nValores nulos:")
print(df.isnull().sum())
print("\nDistribuição da classe quality:")
print(df["quality"].value_counts().sort_index())

#a) PIPELINE FLUXO DE PRÉ-PROCESSAMENTO ATÉ O TREINAMENTO
#=============================================================================

X = df.drop("quality", axis=1)
y = df["quality"]

colunas_numericas   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
colunas_categoricas = X.select_dtypes(include=["object"]).columns.tolist()

print("\nColunas numéricas:", colunas_numericas)
print("Colunas categóricas:", colunas_categoricas)

preprocessador = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ]), colunas_numericas),
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore"))
    ]), colunas_categoricas)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTamanho do treino:", X_train.shape)
print("Tamanho do teste: ", X_test.shape)

modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200,
        random_state=42
    )
}

resultados    = []
melhor_modelo = None
melhor_nome   = None
melhor_f1_macro = -1

for nome, classificador in modelos.items():
    print("\n" + "=" * 60)
    print("Treinando:", nome)
    print("=" * 60)

    pipeline = Pipeline(steps=[
        ("preprocessamento", preprocessador),
        ("classificador",    classificador)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    #b) MÉTRICAS: acurácia global, f1-score e acurácia por classe via matriz
    acuracia   = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("Acurácia global:", round(acuracia, 4))
    print("F1 macro:       ", round(f1_macro, 4))
    print("F1 ponderado:   ", round(f1_weighted, 4))
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))

    matriz  = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    classes = sorted(y.unique())

    print("Matriz de confusão:")
    print(matriz)

    print("\nAcurácia por classe (diagonal da matriz / total da linha):")
    for i, classe in enumerate(classes):
        total_classe = matriz[i].sum()
        acuracia_classe = matriz[i][i] / total_classe if total_classe > 0 else 0
        print(f"  Classe {classe}: {acuracia_classe:.4f}")

    resultados.append({
        "modelo":         nome,
        "acuracia_global": acuracia,
        "f1_macro":       f1_macro,
        "f1_weighted":    f1_weighted
    })

    if f1_macro > melhor_f1_macro:
        melhor_f1_macro = f1_macro
        melhor_modelo   = pipeline
        melhor_nome     = nome

resultados_df = pd.DataFrame(resultados).sort_values(by="f1_macro", ascending=False)

print("\n" + "=" * 60)
print("Comparação final dos modelos")
print("=" * 60)
print(resultados_df.to_string(index=False))

resultados_df.to_csv("wine_model_results.csv", index=False)

#=============================================================================
# c) JUSTIFICATIVA DO MODELO SELECIONADO
#
# O modelo com maior F1 macro é selecionado para implantação. O F1 macro
# pondera igualmente todas as classes, penalizando modelos que acertam bem
# as classes majoritárias mas ignoram as minoritárias, problema comum nesta
# base, onde qualidade 5 e 6 dominam a distribuição.
#
# Random Forest e Extra Trees tendem a superar o AdaBoost aqui porque:
# 1. Suportam class_weight="balanced", compensando o desbalanceamento.
# 2. São menos sensíveis a ruído nas features químicas.
# 3. O AdaBoost, sem suporte nativo a class_weight no sklearn, sofre mais
#nas classes extremas (3, 4, 8).
#=============================================================================

print(f"\nMelhor modelo selecionado: {melhor_nome}")
print(f"F1 macro: {round(melhor_f1_macro, 4)}")

joblib.dump(melhor_modelo, "modelo_wine_quality.pkl")
print("Modelo salvo em: modelo_wine_quality.pkl")


def inferir_qualidade_vinho(dados_vinho):
    modelo_carregado = joblib.load("modelo_wine_quality.pkl")
    vinho_df         = pd.DataFrame([dados_vinho])

    qualidade_prevista = modelo_carregado.predict(vinho_df)[0]

    if hasattr(modelo_carregado.named_steps["classificador"], "predict_proba"):
        probabilidades = modelo_carregado.predict_proba(vinho_df)[0]
        classes        = modelo_carregado.named_steps["classificador"].classes_

        prob_df = pd.DataFrame({
            "quality":       classes,
            "probabilidade": probabilidades
        }).sort_values(by="probabilidade", ascending=False)

        return qualidade_prevista, prob_df

    return qualidade_prevista, None


vinho_desconhecido = {
    "fixed acidity":        7.2,
    "volatile acidity":     0.31,
    "citric acid":          0.32,
    "residual sugar":       6.4,
    "chlorides":            0.038,
    "free sulfur dioxide":  34.0,
    "total sulfur dioxide": 121.0,
    "density":              0.9924,
    "pH":                   3.18,
    "sulphates":            0.47,
    "alcohol":              11.3,
    "wine_type":            "white"
}

classe_prevista, probabilidades = inferir_qualidade_vinho(vinho_desconhecido)

print("\n" + "=" * 60)
print("Inferência — vinho desconhecido")
print("=" * 60)
for chave, valor in vinho_desconhecido.items():
    print(f"  {chave}: {valor}")

print("\nQualidade prevista:", classe_prevista)

if probabilidades is not None:
    print("\nDistribuição de probabilidades:")
    print(probabilidades.to_string(index=False))