FROM python:3.5-slim
WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENV NAME Opentoall

CMD ["python","app.py"]

