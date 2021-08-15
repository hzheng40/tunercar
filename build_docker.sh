#!/usr/bin/env bash

git clone git@git.isis.vanderbilt.edu:SwRI/flight-dynamics-model.git
git clone git@github.com:DARPA-SystemicGenerativeEngineering/swri-uav-pipeline.git
cd swri-uav-pipeline
git submodule init && git submodule update
cd ..
git clone git@github.com:DARPA-SystemicGenerativeEngineering/Conditional-Graph-Completion.git
sudo docker build -t tunercar:gvae -f Dockerfile .