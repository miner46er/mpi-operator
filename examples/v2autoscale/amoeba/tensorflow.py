import os

import horovod.tensorflow as hvd
from kubernetes import client as kube_client, config as kube_config
import time

from .common import *


class AmoebaTensorflow:
    """
    AmoebaTensorflow is a class that handles the autoscaling of an MPIJob running Tensorflow Horovod process.
    """

    def __init__(self,
                 mpijob_name: str = os.getenv("K_MPI_JOB_NAME"),
                 min_worker_size: int = 1,
                 max_worker_size: int = None,
                 delta_worker_size: int = 1,
                 efficiency_lower_limit: float = 0.5):
        """
        Initialize the AmoebaTensorflow class.

        :param mpijob_name: The name of the MPIJob to autoscale.
        :param min_worker_size: The minimum number of workers to scale to.
        :param max_worker_size: The maximum number of workers to scale to.
        :param delta_worker_size: The number of workers to scale.
        :param efficiency_lower_limit: The efficiency threshold for scaling.
        """
        if hvd.rank() != 0:
            return

        if mpijob_name is None:
            raise ValueError("mpijob_name must be specified or environment variable K_MPI_JOB_NAME must be set")

        if min_worker_size < 1:
            raise ValueError("min_worker_size must be >= 1")

        if efficiency_lower_limit <= 0:
            raise ValueError("efficiency_lower_limit must be > 0")

        self.mpijob_name = mpijob_name
        self.min_worker_size = min_worker_size
        self.max_worker_size = max_worker_size
        self.delta_worker_size = delta_worker_size
        self.efficiency_lower_limit = efficiency_lower_limit

        print("""Initializing AmoebaTensorflow
        mpijob_name={},
        min_worker_size={},
        max_worker_size={},
        delta_worker_size={},
        efficiency_threshold={}""".format(
            mpijob_name,
            min_worker_size,
            max_worker_size,
            delta_worker_size,
            efficiency_lower_limit
        ))

        self.__prev_scale_target_size = None
        self.__prev_delta_worker_size = None
        self.__prev_throughput = None
        self.__prev_worker_size = None

        self.__scaling_direction = None
        self.__is_optimal = False

        kube_config.load_incluster_config()
        self.__mpijob_api = kube_client.CustomObjectsApi()

        self.__autoscaling_attempt = 0

    def adjust_scaling(self, cur_throughput: float):
        """
        Adjust the scaling of the MPIJob based on the current throughput.

        :param cur_throughput: The current throughput of the MPIJob.
        """
        if hvd.rank() != 0 or self.__is_optimal:
            return

        cur_size = hvd.size()
        scale_target = cur_size

        if self.__prev_scale_target_size is not None and cur_size != self.__prev_scale_target_size:
            return

        if self.__scaling_direction is None:
            if cur_size == self.max_worker_size:
                # already at max size at first attempt
                self.__scaling_direction = scaling_direction_down
                scale_target = max(cur_size - self.delta_worker_size, self.min_worker_size)
            elif self.__prev_delta_worker_size is not None:
                # this is the second scaling attempt
                cur_throughput_efficiency = self.__get_throughput_efficiency(cur_throughput)
                print("throughput_efficiency: {}".format(cur_throughput_efficiency))
                if cur_throughput_efficiency < self.efficiency_lower_limit:
                    # efficiency is too low, scale down skipping the first size
                    self.__scaling_direction = scaling_direction_down
                    scale_target = max(cur_size - (2 * self.delta_worker_size), self.min_worker_size)
                else:
                    # efficiency is high enough, scale up
                    self.__scaling_direction = scaling_direction_up
                    if self.max_worker_size is None:
                        scale_target = cur_size + self.delta_worker_size
                    else:
                        scale_target = min(cur_size + self.delta_worker_size, self.max_worker_size)
            else:
                # default first scaling attempt, not setting the scaling direction yet
                scale_target = min(cur_size + self.delta_worker_size, self.max_worker_size)

        elif self.__scaling_direction == scaling_direction_up:
            cur_throughput_efficiency = self.__get_throughput_efficiency(cur_throughput)
            print("throughput_efficiency: {}".format(cur_throughput_efficiency))
            if cur_throughput_efficiency < self.efficiency_lower_limit:
                # efficiency is too low, the previous worker size is the optimal worker size
                scale_target = max(cur_size - self.delta_worker_size, self.min_worker_size)
                self.__is_optimal = True
            else:
                # keep scaling up
                if self.max_worker_size is None:
                    scale_target = cur_size + self.delta_worker_size
                else:
                    scale_target = min(cur_size + self.delta_worker_size, self.max_worker_size)

        elif self.__scaling_direction == scaling_direction_down:
            cur_throughput_efficiency = self.__get_throughput_efficiency(cur_throughput)
            print("throughput_efficiency: {}".format(cur_throughput_efficiency))
            if cur_throughput_efficiency < self.efficiency_lower_limit:
                # efficiency is too low, the previous worker size is the optimal worker size
                if self.max_worker_size is None:
                    scale_target = cur_size + self.delta_worker_size
                else:
                    scale_target = min(cur_size + self.delta_worker_size, self.max_worker_size)
                self.__is_optimal = True
            else:
                # keep scaling down
                scale_target = max(cur_size - self.delta_worker_size, self.min_worker_size)

        if scale_target == self.min_worker_size or scale_target == self.max_worker_size or scale_target == cur_size:
            self.__is_optimal = True

        if self.__is_optimal:
            print("found optimal worker size")

        self.__prev_throughput = cur_throughput
        self.__prev_worker_size = cur_size
        self.__prev_delta_worker_size = scale_target - cur_size
        self.__prev_scale_target_size = scale_target
        self.__autoscaling_attempt += 1
        print("autoscaling attempt: {}".format(self.__autoscaling_attempt))
        print("{}\t| scaling to {} worker(s)".format(time.time(), scale_target))
        self.__scale_worker_size(scale_target)

    def __get_throughput_efficiency(self, cur_throughput: float):
        """
        Get the efficiency of the current throughput.

        :param cur_throughput: The current throughput of the MPIJob.
        :return: The efficiency of the current throughput.
        """
        delta_throughput = cur_throughput - self.__prev_throughput
        prev_throughput_per_worker = self.__prev_throughput / self.__prev_worker_size

        return (delta_throughput / self.__prev_delta_worker_size) / prev_throughput_per_worker

    def __scale_worker_size(self, worker_size: int):
        """
        Scale the MPIJob to the given worker size.

        :param worker_size: The number of workers to scale to.
        """
        patch_body = {
            "spec": {
                "mpiReplicaSpecs": {
                    "Worker": {
                        "replicas": worker_size
                    }
                }
            }
        }

        self.__mpijob_api.patch_namespaced_custom_object(
            group=mpijob_group,
            version=mpijob_version,
            namespace=mpijob_namespace,
            plural=mpijob_plural,
            name=self.mpijob_name,
            body=patch_body,
        )
