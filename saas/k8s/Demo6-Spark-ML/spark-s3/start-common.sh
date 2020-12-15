#!/bin/sh
unset SPARK_MASTER_PORT
export HADOOP_HOME=/opt/hadoop
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/hadoop/lib/native
export SPARK_DIST_CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath)
