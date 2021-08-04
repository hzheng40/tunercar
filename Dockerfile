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

FROM rayproject/ray:latest

ARG DEBIAN_FRONTEND="noninteractive"
USER $RAY_UID

RUN sudo apt-get update --fix-missing && \
    sudo apt-get install -y python3-dev python3-pip

RUN sudo apt-get install -y software-properties-common
RUN sudo add-apt-repository ppa:freecad-maintainers/freecad-stable
RUN sudo apt-get update

RUN sudo apt-get install -y git \
                       vim \
                       tmux \
                       automake \
                       build-essential \
                       gfortran \
                       freecad \
                       wget \
                       unzip

RUN pip3 install --upgrade pip

RUN pip3 install numpy>=1.20.2 \
                 scipy \
                 numba \
                 pyyaml \
                 sacred \
                 nevergrad \
                 f90nml==1.3.1 \
                 pandas==1.0.3 \
                 networkx==2.5 \
                 parea==0.1.1

RUN cd /tmp && wget https://github.com/Zolko-123/FreeCAD_Assembly4/archive/master.zip
RUN unzip /tmp/FreeCAD_Assembly4-master.zip -d /lib/freecad/Mod/

RUN mkdir -p $HOME/tunercar/es
RUN mkdir $HOME/swri-uav-pipeline
RUN mkdir $HOME/fdm-wrapper
RUN mkdir $HOME/flight-dynamics-model
COPY ./es $HOME/tunercar/es
COPY ./swri-uav-pipeline $HOME/swri-uav-pipeline
COPY ./fdm-wrapper $HOME/fdm-wrapper
COPY ./flight-dynamics-model $HOME/flight-dynamics-model

RUN cd $HOME/flight-dynamics-model && ./configure && make

ENV PROPELLER_DIR=$HOME/fdm-wrapper/propeller
ENV FDM_EXECUTABLE=$HOME/flight-dynamics-model/bin/new_fdm
ENV CAD_DIR=$HOME/swri-uav-pipeline/uav-cad-models
ENV PATH_TO_FREECAD_LIBDIR=/lib/freecad-python3/lib
# TODO: confirm how to import freecad, having trouble on both macos and ubuntu
WORKDIR $HOME/tunercar/es
ENTRYPOINT ["/bin/bash"]