FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as builder
LABEL stage=builder

RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libgl1-mesa-glx \
		libhdf5-dev \
		libopenblas-dev \
		liblapack-dev \
		libssl-dev \
		mc \
		nano \
		python3-dev \
		python3-pip \
		unzip \
		vim \
		wget \
		&& \
	apt-get clean && \
	apt-get autoremove

RUN pip3 install -U pip
RUN pip3 --no-cache-dir install jupyter
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN git clone https://github.com/Skoltech-CHR/DeepField.git /home/DeepField

WORKDIR "/home/DeepField"

RUN pip3 --no-cache-dir install . 

EXPOSE 8888

ENTRYPOINT ["/bin/bash"]
