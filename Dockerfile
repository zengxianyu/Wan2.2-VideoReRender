FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
#RUN pip install -U torchvision
#RUN pip install -U torchaudio
#RUN pip install opencv-python-headless

RUN apt update && apt install -y ffmpeg git git-lfs
