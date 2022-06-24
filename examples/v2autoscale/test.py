from amoeba import AmoebaTensorflow

from kubernetes import client as kube_client, config as kube_config
from amoeba.common import *

if __name__ == "__main__":
    #x = AmoebaTensorflow("test-mpijob")
    kube_config.load_incluster_config()
    mpijob_api = kube_client.CustomObjectsApi()
    worker_size = 2
    patch_body = {
        "spec": {
            "mpiReplicaSpecs": {
                "Worker": {
                    "replicas": worker_size
                }
            }
        }
    }

    mpijob_api.patch_namespaced_custom_object(
        group=mpijob_group,
        version=mpijob_version,
        namespace=mpijob_namespace,
        plural=mpijob_plural,
        name="tensorflow-horovod-test",
        body=patch_body,
    )

    mpijob_api.patch_namespaced_custom_object(
        group="kubeflow.org",
        namespace="default",
        version="v2beta1",
        plural="mpijobs",
        name="tensorflow-horovod-test",
        body=patch_body,
    )