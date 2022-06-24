FROM horovod/horovod-cpu:latest

# Allow OpenSSH to talk to containers without asking for confirmation
# by disabling StrictHostKeyChecking.
# mpi-operator mounts the .ssh folder from a Secret. For that to work, we need
# to disable UserKnownHostsFile to avoid write permissions.
# Disabling StrictModes avoids directory and files read permission checks.
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

RUN pip install kubernetes mpi4py

COPY tensorflow2_mnist_elastic_test.py /autoscale/
COPY tensorflow2_mnist_elastic_test_modified.py /autoscale/
COPY tensorflow2_keras_mnist_elastic_test.py /autoscale/
COPY tensorflow2_synthetic_benchmark_elastic_custom.py /autoscale/
COPY amoeba /autoscale/amoeba
