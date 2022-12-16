ARG RAPIDS_VERSION=22.12
ARG CUDA_VERSION=11.5
ARG LINUX_VERSION=ubuntu20.04
ARG PYTHON_VERSION=3.9
FROM nvcr.io/nvidia/rapidsai/rapidsai-core:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-base-${LINUX_VERSION}-py${PYTHON_VERSION}

WORKDIR /rapids/
RUN mkdir plotly_census_demo

WORKDIR /rapids/plotly_census_demo
RUN mkdir data
WORKDIR /rapids/plotly_census_demo/data
RUN curl https://rapidsai-data.s3.us-east-2.amazonaws.com/viz-data/total_population_dataset.parquet -o total_population_dataset.parquet

WORKDIR /rapids/plotly_census_demo

COPY . .

RUN mamba env create --name plotly_env --file environment.yml

ENTRYPOINT ["bash","./entrypoint.sh"]