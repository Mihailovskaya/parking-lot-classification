FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3


RUN apt install -y git
RUN apt update\
    && apt install -y build-essential cmake libgtk2.0-dev pkg-config  libswscale-dev libtbb2 libtbb-dev\
    && apt install -y curl
    

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

RUN apt-get install -y python3-matplotlib
COPY files/configs/service/requirements.txt /root/requirements.txt
RUN pip3 install --upgrade pip && \
    python3 -m pip install --no-deps -r /root/requirements.txt

RUN pip3 install pycocotools --no-dependencies


WORKDIR /root/parking_lot_classification

    
RUN cd /\
    && git clone https://github.com/NVIDIA-AI-IOT/jetcam.git\
    && cd jetcam\
    && python3 setup.py install

RUN cd /\
    && git clone https://github.com/fbcotter/pytorch_wavelets\
    && cd pytorch_wavelets\
    && pip install .

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
RUN cd /\
    && ldconfig\
    && git clone https://github.com/JunnYu/mish-cuda\
    && cd mish-cuda\
    && python3 setup.py build install


RUN pip3 install traitlets
RUN apt-get -y install python3-tk

COPY . /root/parking_lot_classification

RUN cd /root/parking_lot_classification\
    && mkdir image\
    && mkdir logs

CMD ./start.sh




    

    

