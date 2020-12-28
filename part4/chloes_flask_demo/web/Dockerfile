FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install -y python-pip3 python3-dev build-essential
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]
