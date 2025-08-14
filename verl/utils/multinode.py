import datetime
import logging
import os
import pickle
import subprocess
import time
from typing import Any, List

import torch.distributed as dist

import utils.globals as rift_globals
from utils.common import aws_copy, get_exp_dir, set_os_env_vars, subprocess_run
from utils.profiler import profiler_decorator

USE_S3_FOR_MULTINODE_COMMUNICATION = True


class _MultinodeCommunicationGLOO:
    def __init__(self):
        if rift_globals.job_env == "local":  # local
            self.master_addr = "localhost"
            self.master_port = 1237
        else:  # scale train
            self.master_addr = os.environ["LEADER_ADDR"]
            self.master_port = os.environ["LEADER_PORT"]

        self.multinode_barrier_count = -1
        self.multinode_barrier_timeout = 2 * 60 * 60  # raise timeout error after 2 hours

        self.process_group_initialized = False
        # self.copy_master_addr = None
        # self.copy_master_port = None

    def _init_process_group(self, timeout: int = 1800):
        logging.info("initializing multinode communication process group!")
        assert not self.process_group_initialized, "process group already initialized!"

        # ### keep copy of current master addr and port
        # self.copy_master_addr = os.environ.get("MASTER_ADDR", None)
        # self.copy_master_port = os.environ.get("MASTER_PORT", None)

        self.process_group_initialized = True
        # set the master addr and port for this process
        set_os_env_vars({"MASTER_ADDR": self.master_addr, "MASTER_PORT": str(self.master_port)})
        # initialize the process group
        dist.init_process_group(
            backend="gloo",
            rank=rift_globals.node_rank,
            world_size=rift_globals.num_nodes,
            timeout=datetime.timedelta(0, timeout),
        )

    def _destroy_process_group(self):
        logging.info("destroying multinode communication process group!")
        assert self.process_group_initialized, "process group not initialized!"
        dist.destroy_process_group()
        self.process_group_initialized = False

        # ### restore the master addr and port
        # if self.copy_master_addr is not None:
        #     set_os_env_vars({"MASTER_ADDR": self.copy_master_addr})
        # if self.copy_master_port is not None:
        #     set_os_env_vars({"MASTER_PORT": self.copy_master_port})

    @profiler_decorator
    def barrier(self):
        """
        Wait for all nodes to reach the barrier
        """
        self.multinode_barrier_count += 1

        logging.info("-" * 60)
        logging.info(
            f"reached multinode barrier! multinode_barrier_count: {self.multinode_barrier_count}"
        )
        logging.info("-" * 60)

        ### have to reinitialize the process group because torch distributed is destroyed after inference/RLXF
        self._init_process_group(timeout=self.multinode_barrier_timeout)  # pseudo barrier
        self._destroy_process_group()

    @profiler_decorator
    def all_gather_data(self, data: List[Any]):
        """
        Gather data from all nodes on all nodes
        """
        if rift_globals.num_nodes == 1:  # no need to gather data if there is only one node
            return data
        else:
            raise NotImplementedError(
                "all_gather_data() is not implemented for multinode communication with GLOO!"
            )

    @profiler_decorator
    def broadcast_model(self, model_path: str):
        """
        Broadcast model to all nodes
        """
        if rift_globals.num_nodes == 1:  # no need to broadcast model if there is only one node
            return

        else:
            raise NotImplementedError(
                "broadcast_model() is not implemented for multinode communication with GLOO!"
            )


class _MultinodeCommunicationS3:
    def __init__(self):
        self.multinode_barrier_count = -1
        self.multinode_barrier_timeout = 2 * 60 * 60  # raise timeout error after 2 hours
        self.multinode_barrier_polling_wait_time = 1  # wait 1 seconds before polling s3 again

        self.multinode_comm_local_dir = (
            f"{get_exp_dir()}/multinode_communication/{rift_globals.job_id}"
        )
        self.multinode_comm_s3_dir = (
            f"{get_exp_dir(get_s3_dir=True)}/multinode_communication/{rift_globals.job_id}"
        )

    @profiler_decorator
    def barrier(self):
        """
        Wait for all nodes to reach the barrier
        """
        self.multinode_barrier_count += 1

        logging.info("-" * 60)
        logging.info(
            f"reached multinode barrier! multinode_barrier_count: {self.multinode_barrier_count}"
        )
        logging.info("-" * 60)

        local_dir = (
            f"{self.multinode_comm_local_dir}/multinode_barrier/{self.multinode_barrier_count}"
        )
        s3_dir = f"{self.multinode_comm_s3_dir}/multinode_barrier/{self.multinode_barrier_count}"
        os.makedirs(local_dir)

        ### make empty local file and copy it to s3 for this rank
        file_name = f"node_{rift_globals.node_rank}.ready"
        open(f"{local_dir}/{file_name}", "a").close()
        aws_copy(f"{local_dir}/{file_name}", f"{s3_dir}/{file_name}")

        ### variables to track the sync status
        num_nodes_ready = 1  # this node is already ready
        sync_complete = False
        num_polls = 0

        ### wait until all nodes have finished writing the file
        start_time = time.time()
        while not sync_complete and time.time() - start_time < self.multinode_barrier_timeout:
            result = subprocess.run(
                f"aws s3 ls {s3_dir}/ | grep .ready | wc -l",
                capture_output=True,
                text=True,
                shell=True,
            )
            num_nodes_ready = int(result.stdout)
            num_polls += 1
            time_elapsed = time.time() - start_time

            if num_nodes_ready == rift_globals.num_nodes:
                sync_complete = True
                break

            if num_polls % 10 == 0 or num_polls == 1:
                logging.info(
                    f"num_nodes_ready: {num_nodes_ready}/{rift_globals.num_nodes}, multinode_barrier waiting!, multinode_barrier_count: {self.multinode_barrier_count}, time elapsed: {time_elapsed:.2f}s"
                )
            time.sleep(self.multinode_barrier_polling_wait_time)

        if sync_complete:
            logging.info(
                f"num_nodes_ready: {num_nodes_ready}/{rift_globals.num_nodes}, multinode_barrier sync complete!, multinode_barrier_count: {self.multinode_barrier_count}, time elapsed: {time_elapsed:.2f}s"
            )
        else:
            logging.error(
                f"num_nodes_ready: {num_nodes_ready}/{rift_globals.num_nodes}, multinode_barrier timed out!, multinode_barrier_count: {self.multinode_barrier_count}, time elapsed: {time_elapsed:.2f}s"
            )
            raise TimeoutError("multinode_barrier timed out!")

    @profiler_decorator
    def all_gather_data(self, data: List[Any]):
        """
        Gather data from all nodes on all nodes
        """
        logging.info("-" * 60)
        logging.info("all_gather_data()")

        if rift_globals.num_nodes == 1:  # no need to gather data if there is only one node
            return data

        assert isinstance(data, list), "data should be a list for all_gather"

        gather_id = self.multinode_barrier_count
        local_dir = f"{self.multinode_comm_local_dir}/all_gather_data/{gather_id}"
        s3_dir = f"{self.multinode_comm_s3_dir}/all_gather_data/{gather_id}"

        os.makedirs(local_dir, exist_ok=True)

        ## save data in local pickle file
        file_name = f"node_{rift_globals.node_rank}.pkl"
        with open(f"{local_dir}/{file_name}", "wb") as f:
            pickle.dump(data, f)
        aws_copy(f"{local_dir}/{file_name}", f"{s3_dir}/{file_name}")

        self.barrier()  # wait for all nodes to copy data to s3

        NUM_RETRIES = 5
        for i in range(NUM_RETRIES):  # weird s5cmd behavior, sometimes fails to copy files
            try:
                # copy files from s3 to local
                aws_copy(s3_dir, local_dir, recursive=True)
                # load all copied files
                data_all = []
                for _rank in range(rift_globals.num_nodes):  # load in order of node rank
                    with open(f"{local_dir}/node_{_rank}.pkl", "rb") as f:
                        data_all.extend(pickle.load(f))
                break
            except Exception as e:
                if i == NUM_RETRIES - 1:
                    raise e
                logging.error(f"error loading all gather files: {e}, trying again ...")
                time.sleep(0.5)
                continue

        # delete local gathered data
        cmd = ["rm", "-rf", local_dir]
        subprocess_run(cmd)

        self.barrier()  # wait for all nodes to finish loading data

        # delete gathered data from s3
        # if rift_globals.node_rank == 0:
        #     aws_delete(s3_dir=s3_dir)

        return data_all

    @profiler_decorator
    def broadcast_model(self, model_path: str):
        """
        Broadcast model to all nodes
        """
        if rift_globals.num_nodes == 1:  # no need to broadcast model if there is only one node
            return

        raise NotImplementedError(
            "broadcast_model() is not implemented for multinode communication with S3!"
        )


### global multinode communication object
if USE_S3_FOR_MULTINODE_COMMUNICATION:
    rift_multinode = _MultinodeCommunicationS3()
else:
    rift_multinode = _MultinodeCommunicationGLOO()


if __name__ == "__main__":

    start_time = time.time()
    rift_multinode_test = _MultinodeCommunicationGLOO()
    init_time = time.time() - start_time
    print(f"init time: {init_time:.2f}s")

    start_time = time.time()
    rift_multinode_test.barrier()
    barrier_time = time.time() - start_time
    print(f"barrier time: {barrier_time:.2f}s")
