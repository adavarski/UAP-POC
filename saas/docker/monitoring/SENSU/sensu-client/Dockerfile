FROM sensu_sensu-server

RUN set -ex && \
    apt-get update && apt-get install -y \
        #netcat-openbsd \
        python3 \
        procps \
        build-essential && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    mkdir -p /usr/local/src && \
    curl https://bootstrap.pypa.io/get-pip.py -o /usr/local/src/get-pip.py && \
    python3 /usr/local/src/get-pip.py --user && \
    ~/.local/bin/pip install requests_unixsocket && \
    # required for sensu-plugins-memory-checks
    gem install vmstat --no-rdoc --no-ri && \
    # install plugins
    sensu-install -p sensu-plugins-disk-checks && \
    sensu-install -p sensu-plugins-memory-checks && \
    sensu-install -p sensu-plugins-uptime-checks && \
    # cleanup
    rm -rf /usr/local/src && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./config/ /tmp/sensu/conf.d/
COPY ./plugins/*.py /etc/sensu/plugins/

COPY ./docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
