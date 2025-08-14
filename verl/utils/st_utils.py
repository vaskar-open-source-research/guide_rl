import json
import logging
import os
import urllib
from enum import Enum

import boto3
from omegaconf import OmegaConf

import utils.globals as rift_globals


def get_env_or_aws_secret(key_name, secretsmanager_location):
    """
    Get the value of an environment variable or AWS secret
    """
    try:
        if os.getenv(key_name):
            return os.getenv(key_name)
        else:
            assert (
                secretsmanager_location is not None
            ), "secretsmanager_location is required if env var is not found"
            session = boto3.Session(region_name="us-west-2")
            client = session.client("secretsmanager")
            response = client.get_secret_value(SecretId=secretsmanager_location)
            secret_data = json.loads(response["SecretString"])
            secret_value = secret_data[key_name]
            return secret_value
    except Exception as e:
        logging.error(
            f"error getting key_name: {key_name} from environment or AWS secrets manager: {e}"
        )
        raise e


####################################################################################
#  Following code is copied from rlxf/common_utils.py
####################################################################################


class Environment(Enum):
    SAGEMAKER = "sagemaker"
    EC2 = "ec2"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


def is_sagemaker_env() -> bool:
    # SAGEMAKER_CHECKPOINT_S3_URI and TRAINING_JOB_ARN are scale-train specific
    # env variables that are set

    # SAGEMAKER_MANAGED_WARMPOOL_CACHE_DIRECTORY comes from
    # https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html but unsure if it is still defined
    # even when warmpool is not used
    return (
        "SAGEMAKER_CHECKPOINT_S3_URI" in os.environ
        or "sagemaker" in os.environ.get("TRAINING_JOB_ARN", "")
        or "SAGEMAKER_MANAGED_WARMPOOL_CACHE_DIRECTORY" in os.environ
    )


def is_ec2_env() -> bool:
    try:
        urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id", timeout=3
        ).read().decode()
        return True
    except (urllib.error.HTTPError, urllib.error.URLError):
        # HTTP error comes from k8s. URLError happens if run outside of AWS (ie laptop)
        return False


def is_kubernetes_env() -> bool:
    return "LEADER_ADDR" in os.environ


def get_current_env() -> Environment:
    if is_sagemaker_env():
        return Environment.SAGEMAKER
    elif is_ec2_env():
        return Environment.EC2
    elif is_kubernetes_env():
        return Environment.KUBERNETES
    else:
        return Environment.LOCAL


def get_job_info():
    """
    Get job info (job_env, job_id, num_nodes, node_rank) for the current job
    """
    ### job running on scale train cluster
    if get_current_env() == Environment.KUBERNETES:
        job_env = "scale_train_cluster"
        job_id = str(os.environ["SCALETRAIN_JOB_ID"])
        num_nodes = int(os.environ["NUM_INSTANCES"])
        node_rank = int(os.environ["JOB_COMPLETION_INDEX"])
        leader_port = os.environ["LEADER_PORT"]
        leader_addr = os.environ["LEADER_ADDR"]

    ### job running locally
    else:
        job_env = "local"
        job_id = (
            "local_job_" + rift_globals.JOB_BEGIN_TIME_STR
        )  # create uniqie job_id by appending timestamp
        num_nodes = 1
        node_rank = 0
        leader_port = 6379
        leader_addr = "localhost"

    job_info = OmegaConf.create(
        {
            "job_env": job_env,
            "job_id": job_id,
            "num_nodes": num_nodes,
            "node_rank": node_rank,
            "leader_port": leader_port,
            "leader_addr": leader_addr,
        }
    )
    return job_info
