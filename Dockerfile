FROM ubuntu

RUN apt update
RUN apt install python3 python3-pip -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y tshark

ENV IOTBASE "/"
ENV PYTHONPATH "/src:$PYTHONPATH"

COPY src src
COPY extras extras
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

# Delete the two lines below to mount these volumes instead
COPY datasets datasets
COPY results results
