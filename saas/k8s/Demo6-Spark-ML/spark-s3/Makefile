all: spark
push: push-spark
.PHONY: push push-spark

TAG = 2.1.0

spark:
	docker build -t davarski/spark-s3 .
	docker tag davarski/spark-s3 davarski/spark-s3:$(TAG)

push-spark: spark
	docker push davarski/spark-s3
	docker push davarski/spark-s3:$(TAG)

clean:
	docker rmi davarski/spark-s3:$(TAG) || :
	docker rmi davarski/spark-s3 || :
