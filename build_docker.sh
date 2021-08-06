#!/usr/bin/env bash

git clone git@git.isis.vanderbilt.edu:SwRI/flight-dynamics-model.git
git clone git@github.com:DARPA-SystemicGenerativeEngineering/swri-uav-pipeline.git
cd swri-uav-pipeline
git submodule init && git submodule update
cd ..
sudo docker build -t tunercar:fdm -f Dockerfile .