FROM debian:stretch-slim

COPY ./config/ /usr/local/src/

ENV CODENAME=stretch \
    SENSU_CONFIG_DIR=/etc/sensu/conf.d \
    SENSU_PLUGIN_DIR=/etc/sensu/plugins \
    SENSU_HOME_DIR=/opt/sensu \
    SENSU_USER=sensu \
    SENSU_PASSWORD=secret

# install redis
RUN set -ex && \
    apt-get update && apt-get -y install \
        ca-certificates \
        redis-server \
        curl \
        wget \
        gnupg2 \
        apt-transport-https && \
    mkdir -p $SENSU_CONFIG_DIR && \
    cp /usr/local/src/redis.json $SENSU_CONFIG_DIR && \
    echo "ULIMIT=65536" >> /etc/default/redis-server

# install sensu
RUN set -ex && \
    cd /usr/local/src && \
    wget -q https://sensu.global.ssl.fastly.net/apt/pubkey.gpg -O- | apt-key add - && \
    echo "deb https://sensu.global.ssl.fastly.net/apt $CODENAME main" | tee /etc/apt/sources.list.d/sensu.list && \
    apt-get update && apt-get install -y sensu && \
    cp /usr/local/src/rabbitmq.json $SENSU_CONFIG_DIR && \
    cp /usr/local/src/transport.json $SENSU_CONFIG_DIR && \
    cp /usr/local/src/api.json $SENSU_CONFIG_DIR && \
    cp /usr/local/src/handlers.json $SENSU_CONFIG_DIR && \
    cp /usr/local/src/state_change_only.json $SENSU_CONFIG_DIR && \
    cp /usr/local/src/slack_handler.json $SENSU_CONFIG_DIR && \
	$SENSU_HOME_DIR/embedded/bin/gem install sensu-plugins-slack

# cleanup
RUN rm -rf /usr/local/src && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 4567

HEALTHCHECK --interval=60s CMD curl --fail http://127.0.0.1:4567/health || exit 1

ENV PATH $SENSU_PLUGIN_DIR:$SENSU_HOME_DIR/embedded/bin:$PATH

COPY ./docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
