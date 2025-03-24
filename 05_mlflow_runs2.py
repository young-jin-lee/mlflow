import mlflow

if __name__=="__main__":
    with mlflow.start_run(run_name="mlflow_runs") as run:

        # Your ML code goes here
        mlflow.log_param("learning_rate", 0.01)

        print("Run ID: ", run.info.run_id)

        print("Run Info: ", run.info)