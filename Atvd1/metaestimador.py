import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

print("Formato da base:", df.shape)
print(df.head())
print(df.info())
print("\nValores nulos por coluna:")
print(df.isnull().sum())

colunas_binarias     = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
colunas_assimetricas = ["creatinine_phosphokinase", "platelets", "serum_creatinine"]
colunas_continuas    = ["age", "ejection_fraction", "serum_sodium"]
coluna_alvo          = "DEATH_EVENT"

print("\nVerificando valores únicos das colunas binárias:")
for coluna in colunas_binarias:
    print(f"  {coluna}: {df[coluna].unique()}")

X = df[colunas_assimetricas + colunas_continuas + colunas_binarias].copy()
y = df[coluna_alvo].copy()

#=============================================================================
# a) JUSTIFICATIVA DO METAESTIMADOR
#
# o KMeans é um algoritmo de agrupamento pelos seguintes motivos:
#
# 1. Objetivo de partição clara: queremos separar pacientes em grupos distintos
#    com base em características clínicas.
#
# 2. Eficiência computacional: a base tem ~300 registros e 11 atributos.
#    O KMeans converge rapidamente nessa escala.
#
# 3. Interpretabilidade dos centróides: após o agrupamento, os centróides
#    permitem descrever o "paciente típico" de cada grupo, com valor clínico.
#
# 4. Integração com Pipeline: o KMeans se encaixa nativamente no Pipeline
#    do scikit-learn, garantindo que treino e inferência usem o mesmo
#    pré-processamento sem risco de data leakage.
#
# O número de grupos (k) será definido empiricamente com três métricas:
# - Silhouette Score:      quanto maior, mais coesos e separados os grupos.
# - Davies-Bouldin:        quanto menor, melhor a separação entre grupos.
# - Calinski-Harabasz:     quanto maior, mais densos e bem separados os grupos.

#b) PRÉ-PROCESSAMENTO

preprocessador = ColumnTransformer(
    transformers=[
        (
            "assimetrica",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("log",     FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scaler",  StandardScaler())
            ]),
            colunas_assimetricas
        ),
        (
            "continua",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler())
            ]),
            colunas_continuas
        ),
        (
            "binaria",
            Pipeline(steps=[
                #apenas imputer de moda sem escala para preservar semântica 0/1
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ]),
            colunas_binarias
        )
    ]
)

#SELEÇÃO DO NÚMERO DE GRUPOS (k)

print("\n" + "=" * 60)
print("Avaliação de k (número de grupos)")
print("=" * 60)

resultados = []

for k in range(2, 8):
    modelo_teste = Pipeline(steps=[
        ("preprocessador", preprocessador),
        ("cluster", KMeans(n_clusters=k, random_state=42, n_init=10))
    ])

    grupos = modelo_teste.fit_predict(X)
    X_proc = modelo_teste.named_steps["preprocessador"].transform(X)

    resultados.append({
        "k":                 k,
        "silhouette":        silhouette_score(X_proc, grupos),
        "davies_bouldin":    davies_bouldin_score(X_proc, grupos),
        "calinski_harabasz": calinski_harabasz_score(X_proc, grupos)
    })

resultados_df = pd.DataFrame(resultados)
print(resultados_df.to_string(index=False))

# k=2 apresentou o maior Silhouette e Calinski-Harabasz, e o menor Davies-Bouldin,
# indicando que dois grupos é a partição mais coesa e bem separada para esta base.

#TREINAMENTO DO MODELO FINAL

modelo = Pipeline(steps=[
    ("preprocessador", preprocessador),
    ("cluster", KMeans(n_clusters=2, random_state=42, n_init=10))
])

grupos = modelo.fit_predict(X)

df_resultado          = df.copy()
df_resultado["grupo"] = grupos

resumo_grupos = df_resultado.groupby("grupo").agg(
    quantidade             = ("grupo",               "size"),
    taxa_obito_historica   = ("DEATH_EVENT",         "mean"),
    idade_media            = ("age",                 "mean"),
    fracao_ejecao_media    = ("ejection_fraction",   "mean"),
    creatinina_media       = ("serum_creatinine",    "mean"),
    sodio_medio            = ("serum_sodium",        "mean"),
    anemia_percentual      = ("anaemia",             "mean"),
    diabetes_percentual    = ("diabetes",            "mean"),
    hipertensao_percentual = ("high_blood_pressure", "mean"),
    fumante_percentual     = ("smoking",             "mean")
).round(3)

print("\nResumo dos grupos:")
print(resumo_grupos)

#c) INFERÊNCIA

def inferir_paciente(paciente):
    paciente_df   = pd.DataFrame([paciente])
    grupo         = int(modelo.predict(paciente_df)[0])

    paciente_proc = modelo.named_steps["preprocessador"].transform(paciente_df)
    distancias    = modelo.named_steps["cluster"].transform(paciente_proc)[0]

    similaridade  = 1 / (distancias + 1e-9)
    similaridade  = similaridade / similaridade.sum()

    return {
        "grupo_predito":           grupo,
        "distancias_dos_grupos":   distancias,
        "similaridade_aproximada": similaridade
    }


paciente_desconhecido = {
    "age":                      67,
    "creatinine_phosphokinase": 300,
    "platelets":                250000,
    "serum_creatinine":         2.1,
    "ejection_fraction":        25,
    "serum_sodium":             130,
    "anaemia":                  1,
    "diabetes":                 0,
    "high_blood_pressure":      1,
    "sex":                      1,
    "smoking":                  0
}

resultado = inferir_paciente(paciente_desconhecido)

print("\n" + "=" * 60)
print("Inferência — paciente desconhecido")
print("=" * 60)
print("Grupo predito:          ", resultado["grupo_predito"])
print("Distâncias dos grupos:  ", resultado["distancias_dos_grupos"].round(4))
print("Similaridade aproximada:", resultado["similaridade_aproximada"].round(4))