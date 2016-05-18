# ecs251-final-project
Final project for ecs251 operating systems

#Installation
## Mac OSX
Install python

    brew install python

Install virtualenv

    sudo pip install virtualenv

Create a new virtualenvironment and activate

    virtualenv venv --python python3
    source venv/bin/activate

Install requirements

    pip install -r requirements.txt


#Launching a Spark cluster
After you've activated your virtual environment you can launch a spark cluster.
You will first need to set up an AWS account and create login credentials. Along
with configuring your user permissions.

    flintrock --config flintrock.config.yml launch ecs251

This will set up our spark cluster. After this we need to login to our spark
cluster to configure it.

    scp -r ecs251-final-project ec2-user@master.public.ip.address
    flintrock --config flintrock.config.yml login ecs251
    cd ecs251-final-project
    ./bootstrap.sh

#Running
## Locally
To run the analysis perform the installation steps then run in python

    python learn.py

This will run the analysis and output desired information at the end of the run

## On Spark cluster
Log in to the spark master, run bootstrap and then run the learner

    ./bootstrap.sh
    python learn.py --with-spark
