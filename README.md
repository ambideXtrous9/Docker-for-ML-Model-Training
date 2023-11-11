# Docker for ML Model Training
 How to use docker to train ML Model

```
sudo docker build -f Dockerfile -t docker_ml . --no-cache

sudo docker run -ti docker_ml /bin/bash -c "cd src && python temp.py"

```
