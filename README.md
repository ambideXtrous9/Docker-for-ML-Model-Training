# Docker for ML Model Training
 How to use docker to train ML Model

Check Docker Images
```
sudo docker images
```

Check Docker Running Images

```
sudo docker ps
```

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
sudo docker rmi <Image_Name1> <Image_Name2>
```

Stop all Docker images

```
sudo docker stop $(sudo docker ps -a -q)
```

Delete all images

```
docker system prune -a
```
