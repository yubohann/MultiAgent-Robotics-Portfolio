FROM flink:1.18.0-scala_2.12-java11

USER root

ARG SPARK_VERSION=3.3.2
ARG SPARK_PACKAGE=spark-${SPARK_VERSION}-bin-hadoop3
ARG SPARK_DIST_URL=https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip procps curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fL "${SPARK_DIST_URL}" -o /tmp/${SPARK_PACKAGE}.tgz \
    && tar -xzf /tmp/${SPARK_PACKAGE}.tgz -C /opt/ \
    && rm -f /tmp/${SPARK_PACKAGE}.tgz

RUN ln -s /opt/${SPARK_PACKAGE} /opt/spark \
    && mkdir -p /opt/spark-apps /opt/spark/work \
    && chmod -R 755 /opt/spark /opt/${SPARK_PACKAGE} \
    && chmod -R 777 /opt/spark-apps /opt/spark/work

ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

WORKDIR /opt/spark-apps
