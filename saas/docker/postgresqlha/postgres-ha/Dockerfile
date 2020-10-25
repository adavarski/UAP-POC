FROM postgres:10.13
ENV PG_MAX_WAL_SENDERS 8
ENV PG_WAL_KEEP_SEGMENTS 8
RUN echo "deb http://deb.debian.org/debian stretch main" > /etc/apt/sources.list.d/debian.list && \
    apt-get update -y && \
    apt-get install -y python3-pip curl python3-psycopg2
RUN pip3 install patroni[zookeeper]
COPY postgres0.yml /postgres0.yml
COPY postgres1.yml /postgres1.yml
CMD ["patroni.py"]
