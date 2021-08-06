# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND="noninteractive"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV PATH="/root/miniconda3/bin:${PATH}"

# apt packages
RUN apt-get update \
    && apt-get install -y git \
                          vim \
                          tmux \
                          automake \
                          build-essential \
                          gfortran \
                          wget \
                          unzip \
                          ffmpeg \
                          libsm6 \
                          libxext6 \
                          gcc \
                          zip \
                          unzip \
    && rm -rf /var/lib/apt/lists/*

# conda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda create -n fcenv freecad networkx=2.5 pandas=1.3.1 -c conda-forge -y
RUN conda init bash
SHELL ["conda", "run", "-n", "fcenv", "/bin/bash", "-c"]

ENV HOME=/home/tunercar
RUN mkdir -p $HOME

# pip packages
RUN pip install numpy>=1.20.2 \
                ray \
                scipy \
                numba \
                pyyaml \
                sacred \
                nevergrad \
                f90nml==1.3.1 \
                parea==0.1.1 \
                matplotlib

# Assembly 4 Bench
RUN cd /tmp && wget https://github.com/Zolko-123/FreeCAD_Assembly4/archive/master.zip
RUN unzip /tmp/master.zip -d $HOME

RUN mkdir -p /tunercar/es
COPY ./es /tunercar/es

COPY ./swri-uav-pipeline $HOME/swri-uav-pipeline
COPY ./flight-dynamics-model $HOME/flight-dynamics-model
RUN sed -i 's/PER3_11X45MR.dat/PER3_11x45MR.dat/g' $HOME/swri-uav-pipeline/uav-design-simulator/uav_simulator/design.py

RUN cd $HOME/flight-dynamics-model && autoreconf -f -i && ./configure && make

ENV PROPELLER_DIR=$HOME/swri-uav-pipeline/uav-design-simulator/propeller
ENV FDM_EXECUTABLE=$HOME/flight-dynamics-model/bin/new_fdm
ENV CAD_DIR=$HOME/swri-uav-pipeline/uav-cad-models
ENV PYTHONPATH=$HOME/swri-uav-pipeline/design-generator:$HOME/swri-uav-pipeline/uav-design-simulator:$HOME/FreeCAD_Assembly4-master

WORKDIR $HOME
RUN conda init bash
ENTRYPOINT ["/bin/bash"]