import logging
import time

JOB_BEGIN_TIME_STR = str(int(time.time()))  # used to create unique job_id for local jobs
DEFAULT_S5CMD_FOR_AWS_COPY = True  # default way to copy files from s3 (`s5cmd cp` OR `aws s3 cp`)

### frequently used utility global variables [DO NOT MODIFY]
# set in load_config()
local_exp_dir = ""
s3_exp_dir = ""
job_id = ""
job_env = ""
num_nodes = 1
node_rank = 0

# set in set_random_seed()
job_random_seed = 0
node_random_seed = 0


def print_all_globals():
    logging.info(f"JOB_BEGIN_TIME_STR: {JOB_BEGIN_TIME_STR}")
    logging.info(f"DEFAULT_S5CMD_FOR_AWS_COPY: {DEFAULT_S5CMD_FOR_AWS_COPY}")
    logging.info(f"local_exp_dir: {local_exp_dir}")
    logging.info(f"s3_exp_dir: {s3_exp_dir}")
    logging.info(f"job_env: {job_env}")
    logging.info(f"job_id: {job_id}")
    logging.info(f"num_nodes: {num_nodes}")
    logging.info(f"node_rank: {node_rank}")
    logging.info(f"job_random_seed: {job_random_seed}")
    logging.info(f"node_random_seed: {node_random_seed}")
