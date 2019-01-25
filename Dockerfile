FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7 
RUN git clone https://github.com/nvidia/apex && cd apex && python setup.py install --cuda_ext --cpp_ext
RUN apt-get update && apt-get install -y sox libsox-dev libsox-fmt-all && rm -rf /var/lib/apt/lists/* && git clone https://github.com/pytorch/audio && cd audio && python setup.py install
RUN pip install wget && git clone --recursive https://github.com/parlance/ctcdecode && cd ctcdecode && python setup.py install
RUN pip install numpy python-levenshtein librosa SoundFile tqdm toml tensorboardX marshmallow==2.15.1 python_speech_features
ADD . /workspace/patter
RUN cd patter &&  python setup.py install

