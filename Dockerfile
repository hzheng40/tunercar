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

# FROM rayproject/ray:latest
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND="noninteractive"
# USER $RAY_UID

RUN apt-get update --fix-missing && \
    apt-get install -y python3-dev python3
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:freecad-maintainers/freecad-stable
# RUN sudo add-apt-repository -y ppa:freecad-maintainers/freecad-daily
RUN apt-get update && apt-get install -y git \
                                              vim \
                                              tmux \
                                              automake \
                                              build-essential \
                                              gfortran \
                                              freecad \
                                              wget \
                                              unzip \
                                              python3-pip

RUN pip3 install --upgrade pip

RUN pip3 install numpy>=1.20.2 \
                 ray \
                 scipy \
                 numba \
                 pyyaml \
                 sacred \
                 nevergrad \
                 f90nml==1.3.1 \
                 pandas==1.0.3 \
                 networkx==2.5 \
                 parea==0.1.1 \
                 matplotlib \
                 pyside6

RUN cd /tmp && wget https://github.com/Zolko-123/FreeCAD_Assembly4/archive/master.zip
RUN unzip /tmp/master.zip -d /usr/lib/freecad-python3/Mod/
# RUN unzip /tmp/master.zip -d /usr/lib/freecad/Mod/
RUN sed -i '23s/.*/from PySide6 import QtGui, QtCore/' /usr/lib/freecad-python3/Mod/FreeCAD_Assembly4-master/libAsm4.py
# RUN sed -i '23s/.*/from PySide6 import QtGui, QtCore/' /usr/lib/freecad/Mod/FreeCAD_Assembly4-master/libAsm4.py

RUN mkdir -p /tunercar/es
RUN mkdir /swri-uav-pipeline
RUN mkdir /fdm-wrapper
RUN mkdir /flight-dynamics-model
COPY ./es /tunercar/es
COPY ./swri-uav-pipeline /swri-uav-pipeline
COPY ./fdm-wrapper /fdm-wrapper
COPY ./flight-dynamics-model /flight-dynamics-model

RUN cd /flight-dynamics-model && ./configure && make

ENV PROPELLER_DIR=/fdm-wrapper/propeller
ENV FDM_EXECUTABLE=/flight-dynamics-model/bin/new_fdm
ENV CAD_DIR=/swri-uav-pipeline/uav-cad-models
# ENV PATH_TO_FREECAD_LIBDIR=/lib/freecad/lib
ENV PYTHONPATH=/swri-uav-pipeline/design-generator:/swri-uav-pipeline/uav-design-simulator:/lib/freecad/lib

RUN sed -i "13s/.*/import FreeCAD/" /swri-uav-pipeline/uav-design-simulator/uav_simulator/assembly/generate_assembly.py
RUN sed -i "15s/.*/#/" /swri-uav-pipeline/uav-design-simulator/uav_simulator/assembly/generate_assembly.py
RUN sed -i "14s/.*/import FreeCAD/" /swri-uav-pipeline/uav-design-simulator/uav_simulator/assembly/compute_metrics.py
RUN sed -i "17s/.*/#/" /swri-uav-pipeline/uav-design-simulator/uav_simulator/assembly/compute_metrics.py

# WORKDIR $HOME/tunercar/es
WORKDIR /
ENTRYPOINT ["/bin/bash"]