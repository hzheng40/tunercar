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

RUN sudo apt-get install -y git \
                       vim \
                       tmux \
                       automake \
                       build-essential \
                       gfortran

RUN pip3 install --upgrade pip

RUN pip3 install numpy \
                 scipy \
                 numba \
                 pyyaml \
                 sacred \
                 nevergrad \
                 f90nml

RUN mkdir -p $HOME/tunercar/es
RUN mkdir $HOME/flight-dynamics-model
RUN mkdir $HOME/fdm-wrapper
COPY ./es $HOME/tunercar/es
COPY ./flight-dynamics-model $HOME/flight-dynamics-model
COPY ./fdm-wrapper $HOME/fdm-wrapper

RUN cd $HOME/flight-dynamics-model && ./configure && make

ENV PROPELLER_DIR=$HOME/fdm-wrapper/propeller
ENV FDM_EXECUTABLE=$HOME/flight-dynamics-model/bin/new_fdm
WORKDIR $HOME/tunercar/es
ENTRYPOINT ["/bin/bash"]