FROM nvidia/cuda:10.1-cudnn7-devel



ENV DEBIAN_FRONTEND noninteractive	
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*	  
RUN ln -sv /usr/bin/python3 /usr/bin/python

#RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py
ENV PATH="/root/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py

RUN pip install --user tensorboard
RUN pip install --user torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN pip install pandas google-cloud-storage cloudml-hypertune

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/gaurav67890/Detectron2 detectron2_repo
	
ENV FORCE_CUDA="1"

ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"


RUN pip install -e /detectron2_repo

ENV FVCORE_CACHE="/tmp"

WORKDIR /detectron2_repo

#COPY split_damages .

#ENTRYPOINT ["python3","trainer.py"]
