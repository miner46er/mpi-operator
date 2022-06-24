from datetime import datetime

from kubernetes import client, config, watch

config.load_kube_config()

v1 = client.CoreV1Api()
w = watch.Watch()

print("Start watching nodes")

for event in w.stream(v1.list_node, watch=False):
    print("NODE_NAME: " + event['object'].metadata.name)  # print the name
    print("TIME: " + str(datetime.now()))  # print the time
    for condition in event['object'].status.conditions:
        if condition.type != "Ready":
            continue

        if condition.status == "True":
            print("Node is ready")
        else:
            print("Node readiness: " + str(condition.status))
        break

    # if (event['object'].status.phase == "Succeeded") and (event['type'] != "DELETED"):
    # do below when condition is met
    #     print ("----> This pod succeeded, do something here!")
    print("---")
