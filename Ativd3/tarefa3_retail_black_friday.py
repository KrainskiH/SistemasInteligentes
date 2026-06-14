import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ARQUIVO_DADOS = "retail_black_friday_sales_100k.csv"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

TARGETS = ["product_category", "payment_method", "age_group"]

#IDs e os próprios alvos são removidos da entrada do modelo
COLUNAS_REMOVIDAS = ["transaction_id", "customer_id", "product_id"] + TARGETS

#TRANSFORMER CUSTOMIZADO — EXTRAÇÃO DE FEATURES DE DATA
#
# purchase_date é uma string de data que, no formato bruto, não pode ser
# usada diretamente por modelos numéricos. Este transformer extrai mês,
# dia e dia da semana como variáveis numéricas, e faz parte do Pipeline
# para garantir que treino e inferência passem pela mesma transformação.
#=============================================================================

class DateFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, date_col="purchase_date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_col in X.columns:
            data = pd.to_datetime(X[self.date_col], errors="coerce")
            X["purchase_month"]     = data.dt.month
            X["purchase_day"]       = data.dt.day
            X["purchase_dayofweek"] = data.dt.dayofweek
            X = X.drop(columns=[self.date_col])
        return X


#CÁLCULO DE MÉTRICAS POR CLASSE (ONE-VS-REST)
#
#Para cada classe, calculamos TP, TN, FP e FN usando a lógica binária:
# Sensibilidade= TP / (TP + FN) taxa de acerto na classe positiva
# Especificidade= TN / (TN + FP) taxa de acerto nas demais classes
#  Acurácia/classe  (TP + TN) / Total
#  F1 = média harmônica entre precisão e sensibilidade


def calcular_metricas_por_classe(y_real, y_predito):
    classes = sorted(pd.unique(pd.Series(list(y_real) + list(y_predito))))
    matriz  = confusion_matrix(y_real, y_predito, labels=classes)
    total   = matriz.sum()

    linhas = []
    for i, classe in enumerate(classes):
        tp = matriz[i, i]
        fn = matriz[i, :].sum() - tp
        fp = matriz[:, i].sum() - tp
        tn = total - tp - fn - fp

        acuracia_classe = (tp + tn) / total if total > 0 else 0
        sensibilidade   = tp / (tp + fn)   if (tp + fn) > 0 else 0
        especificidade  = tn / (tn + fp)   if (tn + fp) > 0 else 0
        precisao        = tp / (tp + fp)   if (tp + fp) > 0 else 0
        f1              = (2 * precisao * sensibilidade) / (precisao + sensibilidade) \
                          if (precisao + sensibilidade) > 0 else 0

        linhas.append({
            "classe":            classe,
            "suporte":           int(tp + fn),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "acuracia_por_classe": acuracia_classe,
            "sensibilidade":     sensibilidade,
            "especificidade":    especificidade,
            "f1_score_classe":   f1
        })

    return pd.DataFrame(linhas), pd.DataFrame(matriz, index=classes, columns=classes)

print("\nCarregando base...")
df = pd.read_csv(ARQUIVO_DADOS)

print("Formato:", df.shape)
print(df.head())
print(df.dtypes)
print("\nValores nulos:")
print(df.isnull().sum())

print("\nDistribuição das classes alvo:")
for alvo in TARGETS:
    print(f"\n{alvo}")
    print(df[alvo].value_counts())


#a)PIPELINE FLUXO DE PRÉ-PROCESSAMENTO ATÉ O TREINAMENTO


X_base = df.drop(columns=COLUNAS_REMOVIDAS)

#Detecta colunas numéricas e categóricas após a transformação de data
X_exemplo           = DateFeatureAdder().transform(X_base.head())
colunas_numericas   = X_exemplo.select_dtypes(include=[np.number]).columns.tolist()
colunas_categoricas = X_exemplo.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nColunas numéricas após transformação de data:", colunas_numericas)
print("Colunas categóricas:", colunas_categoricas)

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessador = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ]), colunas_numericas),
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  onehot)
    ]), colunas_categoricas)
])

modelos_candidatos = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=15
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=40,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=18,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=40,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=18,
        n_jobs=-1
    )
}

resultados_gerais        = []
melhores_modelos         = {}
metricas_melhores_modelos = {}

#TREINAMENTO E AVALIAÇÃO — UM LOOP POR VARIÁVEL ALVO

for alvo in TARGETS:
    print("\n" + "=" * 80)
    print(f"Treinando para: {alvo}")
    print("=" * 80)

    y = df[alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X_base, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    melhor_pipeline  = None
    melhor_nome      = None
    melhor_f1_macro  = -1
    melhor_y_pred    = None
    melhor_y_test    = None

    for nome_modelo, classificador in modelos_candidatos.items():
        pipeline = Pipeline(steps=[
            ("date_features", DateFeatureAdder()),
            ("preprocessador", clone(preprocessador)),
            ("classificador",  clone(classificador))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # b) MÉTRICAS
        acuracia_global    = accuracy_score(y_test, y_pred)
        f1_macro           = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_weighted        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        metricas_classe, _ = calcular_metricas_por_classe(y_test, y_pred)

        sensibilidade_macro  = metricas_classe["sensibilidade"].mean()
        especificidade_macro = metricas_classe["especificidade"].mean()

        resultados_gerais.append({
            "alvo":                alvo,
            "modelo":              nome_modelo,
            "acuracia_global":     acuracia_global,
            "sensibilidade_macro": sensibilidade_macro,
            "especificidade_macro":especificidade_macro,
            "f1_macro":            f1_macro,
            "f1_weighted":         f1_weighted
        })

        print(f"\n{nome_modelo}")
        print("  Acurácia global:     ", round(acuracia_global,     4))
        print("  Sensibilidade macro: ", round(sensibilidade_macro,  4))
        print("  Especificidade macro:", round(especificidade_macro, 4))
        print("  F1 macro:            ", round(f1_macro,            4))
        print("  F1 ponderado:        ", round(f1_weighted,         4))

        if f1_macro > melhor_f1_macro:
            melhor_f1_macro = f1_macro
            melhor_pipeline = pipeline
            melhor_nome     = nome_modelo
            melhor_y_pred   = y_pred
            melhor_y_test   = y_test

    print(f"\nMelhor modelo para {alvo}: {melhor_nome}")

    melhores_modelos[alvo] = {
        "nome_modelo": melhor_nome,
        "pipeline":    melhor_pipeline
    }

    metricas_classe, matriz_confusao = calcular_metricas_por_classe(melhor_y_test, melhor_y_pred)
    metricas_melhores_modelos[alvo]  = metricas_classe

    nome_alvo_arquivo = alvo.replace(" ", "_")
    metricas_classe.to_csv(f"metricas_por_classe_{nome_alvo_arquivo}.csv",  index=False)
    matriz_confusao.to_csv(f"matriz_confusao_{nome_alvo_arquivo}.csv")

    print("\nMatriz de confusão:")
    print(matriz_confusao)
    print("\nMétricas por classe:")
    print(metricas_classe.to_string(index=False))


resultados_df = pd.DataFrame(resultados_gerais)
resultados_df = resultados_df.sort_values(by=["alvo", "f1_macro"], ascending=[True, False])
resultados_df.to_csv("comparacao_modelos_tarefa3.csv", index=False)

print("\n" + "=" * 80)
print("Comparação geral dos modelos")
print("=" * 80)
print(resultados_df.to_string(index=False))

joblib.dump(melhores_modelos, "modelos_tarefa3_retail.pkl")
print("\nModelos salvos em: modelos_tarefa3_retail.pkl")

# =============================================================================
# c) INFERÊNCIA — SISTEMA INTELIGENTE EM FUNCIONAMENTO
#
# A função recebe os dados de uma venda desconhecida e retorna, para cada
# variável alvo (product_category, payment_method, age_group):
# - A classe prevista pelo melhor modelo treinado
# - O grau de certeza (probabilidade da classe vencedora)
# - A distribuição completa de probabilidades por classe
# =============================================================================

def inferir_venda(dados_venda):
    modelos   = joblib.load("modelos_tarefa3_retail.pkl")
    venda_df  = pd.DataFrame([dados_venda])
    resultado = {}

    for alvo, conteudo in modelos.items():
        pipeline        = conteudo["pipeline"]
        nome_modelo     = conteudo["nome_modelo"]
        classe_prevista = pipeline.predict(venda_df)[0]
        probabilidades  = pipeline.predict_proba(venda_df)[0]
        classes         = pipeline.named_steps["classificador"].classes_

        prob_df = pd.DataFrame({
            "classe":        classes,
            "probabilidade": probabilidades
        }).sort_values(by="probabilidade", ascending=False)

        resultado[alvo] = {
            "modelo_usado":   nome_modelo,
            "classe_prevista": classe_prevista,
            "grau_certeza":   float(prob_df.iloc[0]["probabilidade"]),
            "probabilidades": prob_df
        }

    return resultado


venda_desconhecida = {
    "gender":           "Female",
    "city":             "New York",
    "customer_segment": "Premium",
    "original_price":   399.90,
    "discount_pct":     25,
    "final_price":      299.92,
    "quantity":         2,
    "purchase_amount":  599.84,
    "purchase_date":    "2025-11-28",
    "purchase_hour":    21,
    "is_weekend":       0,
    "is_black_friday":  1
}

resultado_inferencia = inferir_venda(venda_desconhecida)

print("\n" + "=" * 80)
print("Inferência — venda desconhecida")
print("=" * 80)
print("\nDados da venda:")
for chave, valor in venda_desconhecida.items():
    print(f"  {chave}: {valor}")

for alvo, info in resultado_inferencia.items():
    print("\n" + "-" * 60)
    print(f"Indicação — {alvo}")
    print("  Modelo usado:    ", info["modelo_usado"])
    print("  Classe prevista: ", info["classe_prevista"])
    print("  Grau de certeza: ", f"{info['grau_certeza']:.2%}")
    print("\n  Distribuição de probabilidades:")
    print(info["probabilidades"].to_string(index=False))