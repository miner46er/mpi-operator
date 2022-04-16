FROM mpioperator/base

RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg2 ca-certificates wget

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O /tmp/key.PUB

# Install Intel oneAPI keys.
RUN apt-get update \
    && apt-key add /tmp/key.PUB \
    && rm /tmp/key.PUB \
    && echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get remove -y gnupg2 ca-certificates wget \
    && apt-get autoremove -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        dnsutils \
        intel-oneapi-mpi \
    && rm -rf /var/lib/apt/lists/*

COPY intel-entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
