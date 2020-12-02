## DataProcessing (Data Lake: MinIO)


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo2-DataProcessing-MinIO/pictures/6-DataLakes_object_processin_pipeline.png" width="800">

### Install MinIO client

```
$ wget https://dl.min.io/client/mc/release/linux-amd64/mc && chmod +x mc && sudo mv mc /usr/local/bin
```

### Configure mc and create buckets 
```
$ mc config host add minio-cluster https://minio.data.davar.com minio minio123 --insecure
mc: Configuration written to `/home/davar/.mc/config.json`. Please update your access credentials.
mc: Successfully created `/home/davar/.mc/share`.
mc: Initialized share uploads `/home/davar/.mc/share/uploads.json` file.
mc: Initialized share downloads `/home/davar/.mc/share/downloads.json` file.
Added `minio-cluster` successfully.
```
Note: --insecure, because of self-signed sertificate (ingress)

Edit /home/davar/.mc/config.json and change "url": "https://minio.data.davar.com" to "url": "http://minio.data.davar.com", because we will not use TLS (`mc --insecure`, if we use https) during this demo.

```
mc mb minio-cluster/upload
mc mb minio-cluster/processed
mc mb minio-cluster/twitter
mc ls minio-cluster
mc config host list
```
### Configure MinIO Events 
```
$ mc admin config set minio-cluster notify_kafka:1 enable=on topic=upload brokers="kafka:9092" sasl_username= sasl_password= sasl_mechanism=plain client_tls_cert= client_tls_key= tls_client_auth=0 sasl=off tls=off tls_skip_verify=off queue_limit=0
$ mc admin config set minio-cluster notify_elasticsearch:1 enable=on format="namespace" index="processed" url="http://elasticsearch:9200" --insecure
$ mc admin config set minio-cluster notify_mqtt:1 enable=on broker="tcp://mqtt:1883" topic=processed password= username= qos=0 keep_alive_interval=0s reconnect_interval=0s queue_dir= queue_limit=0 --insecure

$ mc admin service restart minio-cluster

$ mc admin config get minio-cluster notify_elasticsearch 
notify_elasticsearch enable=off url= format=namespace index= queue_dir= queue_limit=0 
notify_elasticsearch:1 url=http://elasticsearch:9200 format=namespace index=processed queue_dir= queue_limit=0 

$ mc admin config get minio-cluster notify_kafka 
notify_kafka enable=off topic= brokers= sasl_username= sasl_password= sasl_mechanism=plain client_tls_cert= client_tls_key= tls_client_auth=0 sasl=off tls=off tls_skip_verify=off queue_limit=0 queue_dir= version= 
notify_kafka:1 topic=upload brokers=kafka:9092 sasl_username= sasl_password= sasl_mechanism=plain client_tls_cert= client_tls_key= tls_client_auth=0 sasl=off tls=off tls_skip_verify=off queue_limit=0 queue_dir= version= 

$ mc admin config get minio-cluster notify_mqtt 
notify_mqtt:1 broker=tcp://mqtt:1883 topic=processed password= username= qos=0 keep_alive_interval=0s reconnect_interval=0s queue_dir= queue_limit=0 
notify_mqtt enable=off broker= topic= password= username= qos=0 keep_alive_interval=0s reconnect_interval=0s queue_dir= queue_limit=0 


```
### Configure Minio Notifications 
```
$ mc event add minio-cluster/upload arn:minio:sqs::1:kafka --event put --suffix=".csv" 
$ mc event add minio-cluster/processed arn:minio:sqs::1:mqtt --event put --suffix=".gz" 
$ mc event add minio-cluster/processed arn:minio:sqs::1:elasticsearch --event put --suffix=".gz"

$ mc admin service restart minio-cluster ## optional

```
### Test MinIO Events/Notifications: Create test.cvs file and cp to upload bucket and check kafka topic: upload

```
$ echo test > test.csv
$ mc cp test1.csv minio-cluster/upload

kubectl exec -it kafka-client-util bash -n data
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
root@kafka-client-util:/# kafka-topics --zookeeper zookeeper-headless:2181 --list
__confluent.support.metrics
__consumer_offsets
messages
metrics
test
twitter
upload
root@kafka-client-util:/# kafka-console-consumer --bootstrap-server kafka:9092 --topic upload --from-beginning -max-messages 3
{"EventName":"s3:ObjectCreated:Put","Key":"upload/test.csv","Records":[{"eventVersion":"2.0","eventSource":"minio:s3","awsRegion":"","eventTime":"2020-12-02T08:42:59.822Z","eventName":"s3:ObjectCreated:Put","userIdentity":{"principalId":"minio"},"requestParameters":{"accessKey":"minio","region":"","sourceIPAddress":"10.42.0.1"},"responseElements":{"content-length":"0","x-amz-request-id":"164CD9BE9FBE5C85","x-minio-deployment-id":"85826866-b2ff-4c9e-80c6-48d91e742c43","x-minio-origin-endpoint":"http://10.42.0.135:9000"},"s3":{"s3SchemaVersion":"1.0","configurationId":"Config","bucket":{"name":"upload","ownerIdentity":{"principalId":"minio"},"arn":"arn:aws:s3:::upload"},"object":{"key":"test.csv","eTag":"d41d8cd98f00b204e9800998ecf8427e","contentType":"text/csv","userMetadata":{"content-type":"text/csv"},"sequencer":"164CD9BEA0672963"}},"source":{"host":"10.42.0.1","port":"","userAgent":"MinIO (linux; amd64) minio-go/v7.0.6 mc/2020-11-25T23:04:07Z"}}]}
{"EventName":"s3:ObjectCreated:Put","Key":"upload/test1.csv","Records":[{"eventVersion":"2.0","eventSource":"minio:s3","awsRegion":"","eventTime":"2020-12-02T08:44:29.800Z","eventName":"s3:ObjectCreated:Put","userIdentity":{"principalId":"minio"},"requestParameters":{"accessKey":"minio","region":"","sourceIPAddress":"10.42.0.1"},"responseElements":{"content-length":"0","x-amz-request-id":"164CD9D392DE414D","x-minio-deployment-id":"85826866-b2ff-4c9e-80c6-48d91e742c43","x-minio-origin-endpoint":"http://10.42.0.135:9000"},"s3":{"s3SchemaVersion":"1.0","configurationId":"Config","bucket":{"name":"upload","ownerIdentity":{"principalId":"minio"},"arn":"arn:aws:s3:::upload"},"object":{"key":"test1.csv","size":10,"eTag":"71f74d0894d9ce89e22c678f0d8778b2","contentType":"text/csv","userMetadata":{"content-type":"text/csv"},"sequencer":"164CD9D393844E8B"}},"source":{"host":"10.42.0.1","port":"","userAgent":"MinIO (linux; amd64) minio-go/v7.0.6 mc/2020-11-25T23:04:07Z"}}]}
^CProcessed a total of 2 messages

root@kafka-client-util:/# kafka-console-consumer --bootstrap-server kafka:9092 --topic upload --from-beginning -max-messages 1|python -m json.tool
Processed a total of 1 messages
{
    "EventName": "s3:ObjectCreated:Put",
    "Key": "upload/test.csv",
    "Records": [
        {
            "awsRegion": "",
            "eventName": "s3:ObjectCreated:Put",
            "eventSource": "minio:s3",
            "eventTime": "2020-12-02T08:42:59.822Z",
            "eventVersion": "2.0",
            "requestParameters": {
                "accessKey": "minio",
                "region": "",
                "sourceIPAddress": "10.42.0.1"
            },
            "responseElements": {
                "content-length": "0",
                "x-amz-request-id": "164CD9BE9FBE5C85",
                "x-minio-deployment-id": "85826866-b2ff-4c9e-80c6-48d91e742c43",
                "x-minio-origin-endpoint": "http://10.42.0.135:9000"
            },
            "s3": {
                "bucket": {
                    "arn": "arn:aws:s3:::upload",
                    "name": "upload",
                    "ownerIdentity": {
                        "principalId": "minio"
                    }
                },
                "configurationId": "Config",
                "object": {
                    "contentType": "text/csv",
                    "eTag": "d41d8cd98f00b204e9800998ecf8427e",
                    "key": "test.csv",
                    "sequencer": "164CD9BEA0672963",
                    "userMetadata": {
                        "content-type": "text/csv"
                    }
                },
                "s3SchemaVersion": "1.0"
            },
            "source": {
                "host": "10.42.0.1",
                "port": "",
                "userAgent": "MinIO (linux; amd64) minio-go/v7.0.6 mc/2020-11-25T23:04:07Z"
            },
            "userIdentity": {
                "principalId": "minio"
            }
        }
    ]
}

```

### Install/Configure Go  
```
wget https://golang.org/dl/go1.15.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.15.5.linux-amd64.tar.gz

$ tail -n5 ~/.bashrc 

export GOROOT=/usr/local/go
export PATH=${GOROOT}/bin:${PATH}
export GOPATH=$HOME/go
export PATH=${GOPATH}/bin:${PATH}

davar@carbon:~$ source ~/.bashrc 
davar@carbon:~$ go version
go version go1.15.5 linux/amd64

OR logout/login from console:

$ go version
go version go1.15.5 linux/amd64
$ go env


Note As of Go 1.14, Go Modules are ready for production use and
considered the official dependency management system for Go. All
developers are encouraged to use Go Modules for new projects along
with migrating any existing projects.
```
### Create Go compressor app
```
mkdir -p ~/workspace/compressor
cd ~/workspace/compressor/
go mod init github.com/compressor
mkdir cmd
cp GIT_CLONE_LOCATION/composer/compressor.go cmd/
cp GIT_CLONE_LOCATION/composer/Dockerfile .

```
Test app:Execute the compressor application configured with the
buckets upload and processed along with the object upload/donors.csv

```
$ for i in {1..1000000};do echo "test$i" >> donors.csv;done
$ mc cp donors.csv minio-cluster/upload 
donors.csv:                                   10.39 MiB / 10.39 MiB ┃▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┃ 171.63 MiB/s


$ export ENDPOINT=minio.data.davar.com
$ export ACCESS_KEY_ID=minio
$ export ACCESS_KEY_SECRET=minio123
go run ./cmd/compressor.go -f upload -k donors.csv -t processed
$ mc rm minio-cluster/processed/donors.csv.gz

```
### Build/GitHub Push/Test go compressor app

```
docker build -t davarski/compressor:v1.0.0 .
docker login
docker push davarski/compressor:v1.0.0

$ docker run -e ENDPOINT=$ENDPOINT -e ACCESS_KEY_ID=$ACCESS_KEY_ID -e ACCESS_KEY_SECRET=$ACCESS_KEY_SECRET -e ENDPOINT_SSL=false davarski/compressor:v1.0.0 -f=upload -k=donors.csv -t=processed
2020/12/02 18:06:49 Starting download stream upload/donors.csv.
2020/12/02 18:06:49 BEGIN PutObject
2020/12/02 18:06:49 Compress and stream.
2020/12/02 18:06:49 Compressed: 10890288 bytes
2020/12/02 18:06:49 COMPLETE PutObject

```
Check kafka topic and elasticsearch index:

```
$ kubectl exec -it kafka-client-util bash -n data
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
root@kafka-client-util:/# kafka-topics --zookeeper zookeeper-headless:2181 --list
__confluent.support.metrics
__consumer_offsets
messages
metrics
test
twitter
upload

kafka-console-consumer --bootstrap-server kafka:9092 --topic upload --from-beginning -max-messages 1|python -m json.tool
kafka-console-consumer --bootstrap-server kafka:9092 --topic upload --from-beginning -max-messages 5

root@kafka-client-util:/# kafka-console-consumer --bootstrap-server kafka:9092 --topic upload --from-beginning  -max-messages 9 |tail -n1
Processed a total of 9 messages
{"EventName":"s3:ObjectCreated:Put","Key":"upload/donors.csv","Records":[{"eventVersion":"2.0","eventSource":"minio:s3","awsRegion":"","eventTime":"2020-12-02T18:06:16.463Z","eventName":"s3:ObjectCreated:Put","userIdentity":{"principalId":"minio"},"requestParameters":{"accessKey":"minio","region":"","sourceIPAddress":"10.42.0.1"},"responseElements":{"content-length":"0","x-amz-request-id":"164CF87B7F5E766F","x-minio-deployment-id":"85826866-b2ff-4c9e-80c6-48d91e742c43","x-minio-origin-endpoint":"http://10.42.0.160:9000"},"s3":{"s3SchemaVersion":"1.0","configurationId":"Config","bucket":{"name":"upload","ownerIdentity":{"principalId":"minio"},"arn":"arn:aws:s3:::upload"},"object":{"key":"donors.csv","size":10890288,"eTag":"2a787e31b90587c44bfa22d121acc135","contentType":"text/csv","userMetadata":{"content-type":"text/csv"},"sequencer":"164CF87B85323195"}},"source":{"host":"10.42.0.1","port":"","userAgent":"MinIO (linux; amd64) minio-go/v7.0.6 mc/2020-11-25T23:04:07Z"}}]}
root@kafka-client-util:/# kafka-console-consumer --bootstrap-server kafka:9092 --topic upload --from-beginning  -max-messages 9 |tail -n1| python -m json.tool
Processed a total of 9 messages
{
    "EventName": "s3:ObjectCreated:Put",
    "Key": "upload/donors.csv",
    "Records": [
        {
            "awsRegion": "",
            "eventName": "s3:ObjectCreated:Put",
            "eventSource": "minio:s3",
            "eventTime": "2020-12-02T18:06:16.463Z",
            "eventVersion": "2.0",
            "requestParameters": {
                "accessKey": "minio",
                "region": "",
                "sourceIPAddress": "10.42.0.1"
            },
            "responseElements": {
                "content-length": "0",
                "x-amz-request-id": "164CF87B7F5E766F",
                "x-minio-deployment-id": "85826866-b2ff-4c9e-80c6-48d91e742c43",
                "x-minio-origin-endpoint": "http://10.42.0.160:9000"
            },
            "s3": {
                "bucket": {
                    "arn": "arn:aws:s3:::upload",
                    "name": "upload",
                    "ownerIdentity": {
                        "principalId": "minio"
                    }
                },
                "configurationId": "Config",
                "object": {
                    "contentType": "text/csv",
                    "eTag": "2a787e31b90587c44bfa22d121acc135",
                    "key": "donors.csv",
                    "sequencer": "164CF87B85323195",
                    "size": 10890288,
                    "userMetadata": {
                        "content-type": "text/csv"
                    }
                },
                "s3SchemaVersion": "1.0"
            },
            "source": {
                "host": "10.42.0.1",
                "port": "",
                "userAgent": "MinIO (linux; amd64) minio-go/v7.0.6 mc/2020-11-25T23:04:07Z"
            },
            "userIdentity": {
                "principalId": "minio"
            }
        }
    ]
}

```

Open a terminal and port-forward Elasticsearch:
$ kubectl port-forward elasticsearch-0 9200:9200 -n data

The following command returns all records from indexes beginning with processed-:

```
$ curl http://localhost:9200/processed*/_search|python -m json.tool
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  1260  100  1260    0     0   5727      0 --:--:-- --:--:-- --:--:--  5727
{
    "_shards": {
        "failed": 0,
        "skipped": 0,
        "successful": 1,
        "total": 1
    },
    "hits": {
        "hits": [
            {
                "_id": "processed/donors.csv.gz",
                "_index": "processed",
                "_score": 1.0,
                "_source": {
                    "Records": [
                        {
                            "awsRegion": "",
                            "eventName": "s3:ObjectCreated:CompleteMultipartUpload",
                            "eventSource": "minio:s3",
                            "eventTime": "2020-12-02T18:06:49.748Z",
                            "eventVersion": "2.0",
                            "requestParameters": {
                                "accessKey": "minio",
                                "region": "",
                                "sourceIPAddress": "10.42.0.1"
                            },
                            "responseElements": {
                                "content-length": "329",
                                "x-amz-request-id": "164CF88343E00280",
                                "x-minio-deployment-id": "85826866-b2ff-4c9e-80c6-48d91e742c43",
                                "x-minio-origin-endpoint": "http://10.42.0.160:9000"
                            },
                            "s3": {
                                "bucket": {
                                    "arn": "arn:aws:s3:::processed",
                                    "name": "processed",
                                    "ownerIdentity": {
                                        "principalId": "minio"
                                    }
                                },
                                "configurationId": "Config",
                                "object": {
                                    "contentType": "application/octet-stream",
                                    "eTag": "f7a510642b6464bd49b65f1f2cdb9b4d-1",
                                    "key": "donors.csv.gz",
                                    "sequencer": "164CF883451A05B4",
                                    "size": 2333610,
                                    "userMetadata": {
                                        "X-Minio-Internal-actual-size": "2333610",
                                        "content-type": "application/octet-stream"
                                    }
                                },
                                "s3SchemaVersion": "1.0"
                            },
                            "source": {
                                "host": "10.42.0.1",
                                "port": "",
                                "userAgent": "MinIO (linux; amd64) minio-go/v6.0.44"
                            },
                            "userIdentity": {
                                "principalId": "minio"
                            }
                        }
                    ]
                },
                "_type": "event"
            }
        ],
        "max_score": 1.0,
        "total": {
            "relation": "eq",
            "value": 1
        }
    },
    "timed_out": false,
    "took": 213
}
```

### k8s cronjob test
```
$ kubectl create -f cronjob/k8s-cronjob-compress.yaml
```
