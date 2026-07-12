# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


def cargar_datos():
    entrenamiento = pd.read_csv(
        "files/input/train_data.csv.zip",
        compression="zip",
    )

    prueba = pd.read_csv(
        "files/input/test_data.csv.zip",
        compression="zip",
    )

    return entrenamiento, prueba


def limpiar_datos(datos):
    datos = datos.rename(
        columns={"default payment next month": "default"}
    )

    datos = datos.drop(columns=["ID"])

    datos = datos[datos["EDUCATION"] != 0]
    datos = datos[datos["MARRIAGE"] != 0]

    datos["EDUCATION"] = datos["EDUCATION"].apply(
        lambda valor: 4 if valor > 4 else valor
    )

    return datos


def separar_variables(datos):
    caracteristicas = datos.drop(columns=["default"])
    objetivo = datos["default"]

    return caracteristicas, objetivo


def construir_pipeline():

    variables_categoricas = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]
    variables_numericas=["PAY_AMT6","PAY_AMT5", "PAY_AMT4",
                         "PAY_AMT3","PAY_AMT2","PAY_AMT1",
                         "BILL_AMT6","BILL_AMT5","BILL_AMT4","BILL_AMT3",
                         "BILL_AMT2","BILL_AMT1","PAY_6","PAY_5",
                         "PAY_4","PAY_3","PAY_2","PAY_0","LIMIT_BAL","AGE"]
    transformador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                variables_categoricas,
            ),
            (
                "conversion",
                MinMaxScaler(),
                variables_numericas
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", transformador),
            ("seleccion", SelectKBest(score_func=f_classif)),
            ("modelo",LogisticRegression(random_state=0)),
        ]
    )

    return pipeline
def entrenar_modelo(x_train, y_train):

    parametros = {
    'seleccion__k': range(1, 11),
    'modelo__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'modelo__penalty': ['l1', 'l2'],
    'modelo__solver': ['liblinear'],
    "modelo__max_iter": [100, 200]
    }

    grid = GridSearchCV(
        estimator=construir_pipeline(),
        param_grid=parametros,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=7,
        verbose=1,
        refit=True,
    )

    grid.fit(x_train, y_train)

    return grid


def guardar_modelo(modelo):

    os.makedirs("files/models", exist_ok=True)

    with gzip.open(
        "files/models/model.pkl.gz",
        "wb",
    ) as archivo:
        pickle.dump(modelo, archivo, protocol=4)


def calcular_metricas(nombre, reales, predichos):

    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": round(
            precision_score(reales, predichos),
            4,
        ),
        "balanced_accuracy": round(
            balanced_accuracy_score(reales, predichos),
            4,
        ),
        "recall": round(
            recall_score(reales, predichos),
            4,
        ),
        "f1_score": round(
            f1_score(reales, predichos),
            4,
        ),
    }


def calcular_matriz(nombre, reales, predichos):

    matriz = confusion_matrix(reales, predichos)

    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {
            "predicted_0": int(matriz[0][0]),
            "predicted_1": int(matriz[0][1]),
        },
        "true_1": {
            "predicted_0": int(matriz[1][0]),
            "predicted_1": int(matriz[1][1]),
        },
    }


def guardar_resultados(resultados):

    os.makedirs("files/output", exist_ok=True)

    with open(
        "files/output/metrics.json",
        "w",
        encoding="utf-8",
    ) as archivo:

        for resultado in resultados:
            archivo.write(json.dumps(resultado))
            archivo.write("\n")


def main():

    datos_entrenamiento, datos_prueba = cargar_datos()

    datos_entrenamiento = limpiar_datos(datos_entrenamiento)
    datos_prueba = limpiar_datos(datos_prueba)

    x_train, y_train = separar_variables(datos_entrenamiento)
    x_test, y_test = separar_variables(datos_prueba)

    modelo = entrenar_modelo(x_train, y_train)

    guardar_modelo(modelo)

    mejor_modelo = modelo.best_estimator_

    pred_train = mejor_modelo.predict(x_train)
    pred_test = mejor_modelo.predict(x_test)

    resultados = [
        calcular_metricas(
            "train",
            y_train,
            pred_train,
        ),
        calcular_metricas(
            "test",
            y_test,
            pred_test,
        ),
        calcular_matriz(
            "train",
            y_train,
            pred_train,
        ),
        calcular_matriz(
            "test",
            y_test,
            pred_test,
        ),
    ]

    guardar_resultados(resultados)


if __name__ == "__main__":
    main()