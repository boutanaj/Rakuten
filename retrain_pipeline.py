from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import requests
import joblib
import os
import mlflow
import shutil

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    dag_id='retrain_product_classifier',
    default_args=default_args,
    description='DAG de réentraînement automatique du modèle produit',
    schedule_interval='@daily',   # tous les jours (simulation)
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# === Tâche 1 : Charger les nouvelles données dans la base (simple placeholder) ===
def load_new_data():
    # Simuler un ajout dans la base
    print("Simulation : Chargement de nouvelles données dans la base SQLite.")

load_data_task = PythonOperator(
    task_id='load_new_data',
    python_callable=load_new_data,
    dag=dag,
)

# === Tâche 2 : Appeler l'endpoint /training de FastAPI pour réentraîner ===
def call_training_endpoint():
    response = requests.post("http://localhost:8000/training")
    print(f"Réponse API: {response.text}")

training_task = PythonOperator(
    task_id='call_training_api',
    python_callable=call_training_endpoint,
    dag=dag,
)

# === Tâche 3 : Comparer l'ancien et le nouveau modèle, et utiliser MLflow ===
def evaluate_and_register_model():
    old_model_path = "models/classifier.pkl"
    new_model_path = "models/classifier.pkl"  # après réentraînement

    # Simuler une "évaluation" : par exemple ici avec la taille du fichier
    old_size = os.path.getsize(old_model_path)
    new_size = os.path.getsize(new_model_path)

    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Product_Classifier_Experiment")

    with mlflow.start_run(run_name="model_evaluation_run") as run:
        mlflow.log_param("old_model_size", old_size)
        mlflow.log_param("new_model_size", new_size)

        if new_size > old_size:
            mlflow.log_metric("model_improvement", 1)
            mlflow.register_model(
                model_uri="runs:/{}/model".format(run.info.run_id),
                name="ProductClassifierBest"
            )
        else:
            mlflow.log_metric("model_improvement", 0)

evaluate_task = PythonOperator(
    task_id='evaluate_new_model',
    python_callable=evaluate_and_register_model,
    dag=dag,
)

# === Pipeline ===
start = DummyOperator(task_id="start", dag=dag)
end = DummyOperator(task_id="end", dag=dag)

start >> load_data_task >> training_task >> evaluate_task >> end
