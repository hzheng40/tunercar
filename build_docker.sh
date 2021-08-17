#!/usr/bin/env bash

[ ! -d "flight-dynamics-model" ] \
	&& git clone git@git.isis.vanderbilt.edu:SwRI/flight-dynamics-model.git
[ -d "flight-dynamics-model" ] \
	&& cd flight-dynamics-model \
	&& git pull \
	&& cd ..

[ ! -d "swri-uav-pipeline" ] \
	&& mkdir swri-uav-pipeline \
	&& cd swri-uav-pipeline \
	&& git clone git@github.com:DARPA-SystemicGenerativeEngineering/horizon-design-generator.git design-generator \
	&& git clone git@github.com:DARPA-SystemicGenerativeEngineering/uav-cad-models.git \
	&& git clone git@github.com:DARPA-SystemicGenerativeEngineering/uav-design-simulator.git \
	&& git clone git@github.com:DARPA-SystemicGenerativeEngineering/swri-uav-exploration.git \
	&& cd ..

[ -d "swri-uav-pipeline" ] \
	&& cd swri-uav-pipeline \
	&& for d in ./*/ ; do (cd "$d" && git pull); done

cd ..
sudo docker build -t tunercar:fdm -f Dockerfile .