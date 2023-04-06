FROM ubuntu

RUN apt-get update
RUN apt-get upgrade
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

WORKDIR /home

COPY GGMS GGMS
COPY algo_test.ipynb algo_test.ipynb
COPY algo_density_test.ipynb algo_density_test.ipynb
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

CMD jupyter notebook --allow-root --ip 0.0.0.0 --no-browser