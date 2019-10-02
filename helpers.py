#!/usr/bin/env python

import json, requests
from pyspark.sql import SparkSession


def load_json_from_url(url):
    """Collects json observations from an url and parse them into a valid pyspark.sql.dataframe.DataFrame

    :url: A url containing a json file
    :returns: A pyspark.sql.dataframe.DataFrame object
    """
    # generamos una consulta  a la URL
    tmp_requests = requests.get(url)
    # leemos y convertimos cada registro a un registro json válido
    tmp_json_load = [json.loads(line) for line in tmp_requests.iter_lines()]
    return tmp_json_load

def get_als_factors_information(model, n_users = 3, n_items = 10):
    """Report summary on factor loadings at item and user level

    :model: An pyspark.ml.recommendation.ALSModel object.
    :n_users: An integer. Quantity of users to report.
    :n_items: An integer. Quantity of items to report.
    :returns: A printed summary.

    """
    tmp_user_factors, tmp_item_factors = model.userFactors, model.itemFactors
    print(f"Número de usuarios en entrenamiento: {tmp_user_factors.count()}")
    print(f"Producto punto para los primeros {n_users} usuarios:")

    for i in tmp_user_factors.take(n_users):
        print(f"Usuario: {i.id} -> Producto punto: {[round(j, 3) for j in i.features]}" )

    print("\n")
    print(f"Número de items en entrenamiento: {tmp_item_factors.count()}")
    print(f"Producto punto para los primeros {n_items} items:")

    for i in tmp_item_factors.take(n_items):
        print(f"Item: {i.id} -> Producto punto: {[round(j, 3) for j in i.features]}")


def report_reg_metrics(metrics):
    """Report metrics from a RegressionMetrics object

    :metrics: a RegressionMetrics object
    :returns: a printed report.

    """
    print(f"Varianza Explicada: {round(metrics.explainedVariance, 3)}")
    print(f"Error cuadrático promedio: {round(metrics.meanSquaredError, 3)}")
    print(f"Error absoluto promedio: {round(metrics.meanAbsoluteError, 3)}")
    print(f"Raíz del error cuadrático promedio: {round(metrics.rootMeanSquaredError, 3)}")
