#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
sudo docker run -it --rm --shm-size=300gb -v $SCRIPT_DIR/es:/home/tunercar/tunercar/es -v $SCRIPT_DIR/swri-uav-pipeline:/home/tunercar/swri-uav-pipeline tunercar:fdm
