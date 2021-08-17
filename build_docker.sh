#!/usr/bin/env bash

git clone git@git.isis.vanderbilt.edu:SwRI/flight-dynamics-model.git
#git clone git@github.com:DARPA-SystemicGenerativeEngineering/swri-uav-pipeline.git
#cd swri-uav-pipeline
#git submodule init && git submodule update

mkdir swri-uav-pipeline
cd swri-uav-pipeline
git clone git@github.com:DARPA-SystemicGenerativeEngineering/horizon-design-generator.git design-generator
git clone git@github.com:DARPA-SystemicGenerativeEngineering/uav-cad-models.git
git clone git@github.com:DARPA-SystemicGenerativeEngineering/uav-design-simulator.git
git clone git@github.com:DARPA-SystemicGenerativeEngineering/swri-uav-exploration.git
cd ..
sudo docker build -t tunercar:fdm -f Dockerfile .
