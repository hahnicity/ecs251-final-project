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

This will set up our spark cluster

#Running

To run the analysis perform the installation steps then run in python

    python learn.py

This will run the analysis and output desired information at the end of the run
