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

    flintrock launch test-cluster \
        --num-slaves 5 \
        --spark-version 1.6.1 \
        --ec2-key-name key-pair-name \
        --ec2-identity-file /path/to/identity.pem \
        --ec2-ami ami-d1315fb1 \
        --ec2-user ec2-user \
        --ec2-region us-west-1 \
        --ec2-instance-type t2.micro

This will set up an EC2 spark cluster with 5 slaves.

#Running

To run the analysis perform the installation steps then run in python

    python learn.py

This will run the analysis and output desired information at the end of the run
