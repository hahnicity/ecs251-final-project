#!/bin/bash

# bootstrap the installation of a spark master on rhel/fedora

sudo yum install gcc-c++ gcc
sudo yum install python34 python-pip python-virtualenv python34-devel
sudo yum install atlas-devel lapack-devel blas-devel libgfortran libyaml-devel

virtualenv --python python34 venv
source venv/bin/activate
pip install numpy
pip install -r requirements.txt
