FROM python:slim-buster

RUN apt-get update && apt-get install -y sqlite
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
ENV PORT 5000

ENTRYPOINT ["mlflow"]
