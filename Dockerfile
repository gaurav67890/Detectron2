FROM nvidia/cuda:10.1-cudnn7-devel



ENV DEBIAN_FRONTEND noninteractive	
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*	  
RUN ln -sv /usr/bin/python3 /usr/bin/python
ENV PATH="/root/.local/bin:${PATH}"


RUN apt-get update && apt-get install -y curl zip
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

COPY credentials.json /etc/
ENV GOOGLE_APPLICATION_CREDENTIALS="/etc/credentials.json"
RUN gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py

RUN pip install --user tensorboard
RUN pip install --user torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
RUN pip install pandas google-cloud-storage cloudml-hypertune

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/gaurav67890/Detectron2 detectron2_repo -b feat/AICAR-325-hyperparameter-tuning-scratch
	
ENV FORCE_CUDA="1"

ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"


RUN pip install -e /detectron2_repo

ENV FVCORE_CACHE="/tmp"

WORKDIR /detectron2_repo

#RUN gsutil cp gs://hptuning/split_damages.zip .
#RUN unzip split_damages.zip
#COPY split_damages .

ENTRYPOINT ["python3","trainer-main.py","--damage_name","crack","max_iter","600","check_period","50"]
