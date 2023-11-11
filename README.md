# Docker for ML Model Training
 How to use docker to train ML Model

Build the Docker Image

```
sudo docker build -f Dockerfile -t docker_ml . --no-cache
```

Run the Docker Image

```
sudo docker run -ti docker_ml /bin/bash -c "cd src && python temp.py"

```

Remove one or more specific images

```
sudo docker rmi Image Image
```

Stop all Docker images

```
sudo docker stop $(sudo docker ps -a -q)
```

Delete all images

```
docker system prune -a
```
