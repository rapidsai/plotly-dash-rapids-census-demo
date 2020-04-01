ARG CUDA_VERSION=9.2
ARG LINUX_VERSION=ubuntu16.04
FROM rapidsai/rapidsai:cuda10.2-runtime-ubuntu16.04
RUN pwd
RUN source activate rapids

WORKDIR /rapids/
RUN mkdir covid_demo

WORKDIR /rapids/covid_demo

COPY . .

RUN conda env update --name rapids --file environment_for_docker.yml