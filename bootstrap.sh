#!/bin/bash

set -o allexport

# bootstrap the installation of a spark master on rhel/fedora

sudo yum install gcc-c++ gcc
sudo yum install python python-pip python-virtualenv python-devel
sudo yum install atlas-devel lapack-devel blas-devel libgfortran libyaml-devel

export SPARK_HOME=/home/ec2-user/spark
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH

virtualenv venv
source venv/bin/activate
pip install numpy
pip install -r requirements.txt
