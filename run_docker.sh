#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
sudo docker run -it --rm --shm-size=10.24gb -v $SCRIPT_DIR/es/quad_fdm_runs:/home/ray/tunercar/es/quad_fdm_runs tunercar:fdm