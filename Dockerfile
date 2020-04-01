ARG CUDA_VERSION=10.2
ARG LINUX_VERSION=ubuntu16.04
FROM rapidsai/rapidsai:cuda${CUDA_VERSION}-runtime-${LINUX_VERSION}

WORKDIR /rapids/
RUN mkdir covid_demo

WORKDIR /rapids/covid_demo
RUN mkdir data
WORKDIR /rapids/covid_demo/data
RUN curl https://s3.us-east-2.amazonaws.com/rapidsai-data/viz-data/acs2018_county_population.parquet.tar.gz -o acs2018_county_population.parquet.tar.gz
RUN curl https://s3.us-east-2.amazonaws.com/rapidsai-data/viz-data/census_data_minimized.parquet.tar.gz -o census_data_minimized.parquet.tar.gz
RUN tar -xvzf acs2018_county_population.parquet.tar.gz
RUN tar -xvzf census_data_minimized.parquet.tar.gz


WORKDIR /rapids/covid_demo

COPY . .

RUN source activate rapids && conda install -c conda-forge --file environment_for_docker.yml && pip install dash-dangerously-set-inner-html

ENTRYPOINT ["bash","./entrypoint.sh"]
