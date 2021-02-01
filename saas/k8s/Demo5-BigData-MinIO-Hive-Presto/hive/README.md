Build Hive image (with Minio:S3)
```
# Download Apache Hive
curl -L http://mirror.cc.columbia.edu/pub/software/apache/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz -o ./src/apache-hive-3.1.2-bin.tar.gz

# Download Apache Hadoop
curl -L http://archive.apache.org/dist/hadoop/common/hadoop-3.1.2/hadoop-3.1.2.tar.gz -o ./src/hadoop-3.1.2.tar.gz

# Uncompress both
tar -xzvf ./src/apache-hive-3.1.2-bin.tar.gz -C ./src
tar -xzvf ./src/hadoop-3.1.2.tar.gz -C ./src

# Add Jars
./deps.sh

# create container
docker build -t davarski-hive-s3m:3.1.2 .
docker tag davarski-hive-s3m:3.1.2 davarski/hive-s3m:3.1.2-1.0.0
docker login
docker push davarski/hive-s3m:3.1.2-1.0.0
```
