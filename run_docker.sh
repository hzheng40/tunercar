#!/usr/bin/env bash

sudo docker run -it --rm --shm-size=10.24gb -v $HOME/tunercar_siemens/es/quad_fdm_runs:/home/ray/tunercar/es/quad_fdm_runs tunercar:fdm