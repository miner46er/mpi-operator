from kubernetes import client, config, watch
from datetime import datetime

config.load_kube_config()

v1 = client.CoreV1Api()
w = watch.Watch()

print("Start watching pods")

for event in w.stream(v1.list_namespaced_pod, namespace="default", watch=False):
    print("POD_NAME: " + event['object'].metadata.name)  # print the name
    print("TIME: " + str(datetime.now()))  # print the time
    print("PHASE: " + event['object'].status.phase)  # print the status of the pod

    if (event['object'].status.phase == "Succeeded") and (event['type'] != "DELETED"):  # do below when condition is met
        print("----> This pod succeeded!")
    print("---")