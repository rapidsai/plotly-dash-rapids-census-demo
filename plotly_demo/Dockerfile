ARG CUDA_VERSION=10.2
ARG LINUX_VERSION=ubuntu16.04
FROM rapidsai/rapidsai:cuda${CUDA_VERSION}-runtime-${LINUX_VERSION}

WORKDIR /rapids/
RUN mkdir covid_demo

WORKDIR /rapids/covid_demo

COPY . .

RUN source activate rapids && conda list

RUN source activate rapids && conda install -c conda-forge --file environment_for_docker.yml && pip install dash-dangerously-set-inner-html

ENTRYPOINT ["bash","./entrypoint.sh"]
