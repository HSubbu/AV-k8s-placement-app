# AV-k8s-placement-app
project folder for ML model deployment using Kubernetes 



This is repo for project for demonstration of deployment of ML model (containerized) using Kubernetes(k8s)

The Docker image is uploaded on https://hub.docker.com . The image can be pulled using

$ docker pull subbu0319/placement-app

The details of folder/files in the project folder is as follows :

 (a) Notebooks folder - jupyter noteboook for data cleaning , model building with model and Dictvectorizer(encoding of cat columns) as output

 (b) Data - contains train and test data .

 (c) predict.py - python script to prodice an API for model prediction using flask framework

 (d) predict-test.py - python test script with test features for testing the model prediction . It uses requests libary to access REST API.
                       Please keep in mind to enter correct URL in the script for testing the application (depends on loca/rremote testing)

  (e) Pipfile and Pipfile.lock - pipenv generated dependencies files

  (f) Dockerfile - for building Docker image

  (g) kubeconfig folder - contains deployment.yaml and service.yaml for deploying the container on k8s
