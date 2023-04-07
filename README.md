# GGMS
Framework for experimenting with Gaussian Graphical Models.

Build:
```
docker build -t tag .
```

Run:
```
docker run -p 8888:8888 tag
```

Jupyter notebook host will start.

# Dockerfile description

step 1. Install Ubuntu with Python3 and Pip
```
FROM ubuntu

RUN apt-get update
RUN apt-get upgrade
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
```

step 2. Copy files to container
```
WORKDIR /home

COPY . .
```

step 3. Install Python libraries
```
RUN pip3 install -r requirements.txt
```

step 4. Run Jupyter Notebook host
```
CMD jupyter notebook --allow-root --ip 0.0.0.0 --no-browser
```
