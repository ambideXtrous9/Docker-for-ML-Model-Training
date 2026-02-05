# Docker for ML Model Training

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=docker,py,git,pytorch," />
  </a>
</p>

A comprehensive cheat sheet for managing Docker containers, images, and compose environments for Machine Learning workflows.

## 1. Image Operations

**Build Docker Image**
```bash
# Basic build
sudo docker build -t docker_ml .

# Build without cache (force rebuild)
sudo docker build -f Dockerfile -t docker_ml . --no-cache 
```

**List Images**
```bash
sudo docker images
```

**Remove Images**
```bash
# Remove specific image(s)
sudo docker rmi <Image_Name_or_ID>

# Remove all unused images
sudo docker image prune -a
```

## 2. Container Operations

**Run Containers**
```bash
# Run and open interactive terminal (bash)
sudo docker run -it docker_ml /bin/bash

# Run specific command inside container
sudo docker run -it docker_ml /bin/bash -c "cd src && python temp.py"

# Run in background (detached mode)
sudo docker run -d -p 8080:8080 --name my_ml_container docker_ml
```

**Manage Containers**
```bash
# List running containers
sudo docker ps

# List all containers (including stopped ones)
sudo docker ps -a

# Stop a specific container
sudo docker stop <Container_ID_or_Name>

# Stop ALL running containers
sudo docker stop $(sudo docker ps -a -q)

# Remove a specific container
sudo docker rm <Container_ID_or_Name>
```

**Interactive Access**
```bash
# Access shell of a running container
sudo docker exec -it <Container_ID_or_Name> /bin/bash
```

## 3. Docker Compose Workflow

**Start Services**
```bash
# Start and build images
docker compose up --build

# Start in detached mode (background)
docker compose up -d
```

**Stop Services**
```bash
# Stop and remove containers, networks
docker compose down

# Stop specific environment file
docker compose -f docker-compose-development.yml down
```

**View Compose Logs**
```bash
docker compose logs -f
```

## 4. specific project commands

```
# 1. Stop the running containers
docker compose -f docker-compose-development.yml down

# 2. (Optional) Remove the old containers completely to ensure a clean start
docker compose -f docker-compose-development.yml rm -f

# 3. Start the containers with the updated configuration
docker compose -f docker-compose-development.yml up -d

# 4. Verify the containers are running
docker compose -f docker-compose-development.yml ps

# 5. (Optional) Check the logs to ensure everything started correctly
docker compose -f docker-compose-development.yml logs -f
```

**Check Logs for Backend**
```bash
sudo docker logs -f recall-riskscore-backend
```

## 5. Debugging & Maintenance

**Logs & monitoring**
```bash
# Follow log output of a specific container
sudo docker logs -f <Container_ID_or_Name>

# View resource usage stats (CPU/Memory) - Critical for ML
sudo docker stats
```

**Inspection**
```bash
# Inspect container details (JSON format)
sudo docker inspect <Container_ID_or_Name>
```

**System Cleanup**
```bash
# Remove unused images, containers, and networks to free up space
docker system prune -f

# Deep clean (including unused images)
docker system prune -a
```
