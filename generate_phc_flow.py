#!/usr/bin/env python3
from pathlib import Path
from decouple import config,Csv
import yaml
from prefect.run_configs import KubernetesRun
from prefect.storage import GCS
from prefect import Flow, task
from prefect.tasks.shell import ShellTask
from prefect.tasks.secrets import EnvVarSecret

FLOW_STORAGE_BUCKET = config('FLOW_STORAGE_BUCKET')
FLOW_BUCKET_PREFIX = config('FLOW_BUCKET_PREFIX', default='covid-data-model')
JOB_TEMPLATE_PATH = config('JOB_TEMPLATE_PATH', default='/etc/config/templates')
PREFECT_PROJECT = config('PREFECT_PROJECT', default='can-scrape')
KUBERNETES_RUN_LABELS = config('KUBERNETES_RUN_LABELS', cast=Csv(), default='kubernetes')
SERVICE_ACCOUNT_NAME = config('SERVICE_ACCOUNT_NAME')

def get_template(template_name: str, flow_name: str) -> str:
    # Retrieve the template from the RELEASE-NAME-job-templates mounted ConfigMap and do a string replacement on the flow placeholder
    template_path = Path(JOB_TEMPLATE_PATH, template_name).with_suffix(".yaml")
    assert template_path.exists(), f"Job Template file {template_path} does not exist!"
    template_data = template_path.read_text().replace("PREFECT_FLOW_PLACEHOLDER", flow_name)
    return yaml.safe_load(template_data)

@task
def get_env_dict(keys: list) -> dict:
    env_dict = {}
    for k in keys:
        env_dict[k] = EnvVarSecret(k, cast=str).run()
    return env_dict

def main():
    flow_name = "CovidDataModel"
    with Flow(
        name=flow_name,
        schedule=None,
        storage=GCS(bucket=FLOW_STORAGE_BUCKET, key=f"{FLOW_BUCKET_PREFIX}/{flow_name}"),
        run_config=KubernetesRun(
            labels=KUBERNETES_RUN_LABELS,
            service_account_name=SERVICE_ACCOUNT_NAME,
            job_template=get_template("cdm", flow_name),
        )
    ) as f:
        keys = [
            "GCS_STORAGE_BUCKET", "GCS_BUCKET_PREFIX", "OPENBLAS_NUM_THREADS",
            "COVID_MODEL_CORES", "DATA_OUTPUT_DIR", "UPLOAD_FILE_FILTER", "GCS_PARQUET_PATH"
        ]
        env_dict = get_env_dict(keys)
        task = ShellTask(command="/covid-data-model/phc.sh", return_all=True, log_stderr=True, stream_output=True)
        output = task(env=env_dict)

    f.register(project_name=PREFECT_PROJECT)

if __name__ == "__main__":
    main()
