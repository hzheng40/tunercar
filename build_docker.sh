#!/usr/bin/env bash

git clone git@git.isis.vanderbilt.edu:SwRI/flight-dynamics-model.git
git clone git@github.com:DARPA-SystemicGenerativeEngineering/fdm-wrapper.git
sudo docker build -t tunercar:fdm -f Dockerfile .