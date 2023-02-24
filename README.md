# Plotly-Dash + RAPIDS | Census 2020 Visualization

![pr1](https://user-images.githubusercontent.com/35873124/189301695-328af0cc-1878-408d-ba01-bdbc61550628.png)

## Charts

1. Map chart shows the total population points for chosen view and selected area
2. Top counties bar show the top 15 counties for chosen view and selected area
3. Bottom counties bar show the bottom 15 counties for chosen view and selected area
4. Race Distribution shows distribution of individual races across blocks for chosen view and selected area

Cross-filtering is enabled to link all the four charts using box-select tool

## Data-Selection Views

The demo consists of six views and all views are calculated at a block level

- Total Population view shows total Census 2020 population.
- Migrating In view shows net inward decennial migration.
- Stationary view shows population that were stationary.
- Migrating Out view shows net outward decennial migration.
- Net Migration view shows total decennial migration. Points are colored into three categories - migrating in, stationary, migrating out
- Population with Race shows total Census 2020 population colored into seven race categories - White alone, African American alone, American Indian alone, Asian alone, Native Hawaiian alone, Other Race alone, Two or More races.

## Installation and Run Steps

## Base Layer Setup

The visualization uses a Mapbox base layer that requires an access token. Create one for free [here on mapbox](https://www.mapbox.com/help/define-access-token/). Go to the demo root directory's `plotly_demo` folder and create a token file named `.mapbox_token`. Copy your token contents into the file.

**NOTE:** Installation may fail without the token.

## Data

There is 1 main dataset:

- [Total Population Dataset](https://data.rapids.ai/viz-data/net_migration_dataset.parquet) ; Consists of Census 2020 total population with decennial migration from Census 2010 at a block level.
- [Net Migration Dataset](https://data.rapids.ai/viz-data/net_migration_dataset.parquet) ; Net migration from Census 2010 at a block level.

For more information on how the Census 2020 and 2010 Migration data was prepared to show individual points, refer to the `/data_prep_total_population` folder.

### Conda Env

Verify the following arguments in the `environment.yml` match your system(easy way to check `nvidia-smi`):

cudatoolkit: Version used is `11.5`

```bash
# setup conda environment
conda env create --name plotly_env --file environment.yml
source activate plotly_env

# run and access single GPU version
cd plotly_demo
python app.py

# run and access multi GPU version, run `python dask_app.py --help for args info`
# if --cuda_visible_devices argument is not passed, all the available GPUs are used
cd plotly_demo
python dask_app.py --cuda_visible_devices=0,1
```

### Docker

Verify the following arguments in the Dockerfile match your system:

1. CUDA_VERSION: Supported versions are `11.0+`
2. LINUX_VERSION: Supported OS values are `ubuntu16.04, ubuntu18.04, centos7`

The most up to date OS and CUDA versions supported can be found here: [RAPIDS requirements](https://rapids.ai/start.html#req)

```bash
# build
docker build -t plotly_demo .

# run and access single GPU version via: http://localhost:8050 / http://ip_address:8050 / http://0.0.0.0:8050
docker run --gpus all --name single_gpu -p 8050:8050 plotly_demo

# run and access multi GPU version via: http://localhost:8050 / http://ip_address:8050 / http://0.0.0.0:8050
# Use `--gpus all` to use all the available GPUs
docker run --gpus '"device=0,1"' --name multi_gpu -p 8050:8050 plotly_demo dask_app
```

## Requirements

### CUDA/GPU requirements

- CUDA 11.0+
- NVIDIA driver 450.80.02+
- Pascal architecture or better (Compute Capability >=6.0)

> Recommended Memory: NVIDIA GPU with at least 32GB of memory(or 2 GPUs with equivalent GPU memory when running dask version), and at least 32GB of system memory.

### OS requirements

See the [Rapids System Requirements section](https://rapids.ai/start.html#requirements) for information on compatible OS.

## Dependencies

- python=3.9
- cudatoolkit=11.5
- rapids=22.08
- dash=2.5.1
- jupyterlab=3.4.3
- dash-html-components=2.0.0
- dash-core-components=2.0.0
- dash-daq=0.5.0
- dash_bootstrap_components=1.2.0

## FAQ and Known Issues

**What hardware do I need to run this locally?** To run you need an NVIDIA GPU with at least 32GB of memory(or 2 GPUs with equivalent GPU memory when running dask version), at least 32GB of system memory.

**How did you compute migration?** Migration was computed by comparing the block level population for census 2010 and 2020

**How did you compare population having block level boundary changes?** [Relationship Files](https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.html#t10t20) provides the 2010 Census Tabulation Block to 2020 Census Tabulation Block Relationship Files. Block relationships may be one-to-one, many-to-one, one-to-many, or many-to-many. Population count was computed in proportion to take into account the division and collation of blocks across 2010 and 2020.

**How did you determine race?** Race for stationary and inward migration individuals was randomly assigned within a block but they add up accurately at the block level. However, due to how data is anonymized, race for outward migration population could not be calculated.

**How did you get individual point locations?** The population density points are randomly placed within a census block and associated to match distribution counts at a census block level.

**How are the population and distributions filtered?** Use the box select tool icon for the map or click and drag for the bar charts.

**Why is the population data from 2010 and 2020?** Only census data is recorded on a block level, which provides the highest resolution population distributions available. For more details on census boundaries refer to the [TIGERweb app](https://tigerweb.geo.census.gov/tigerwebmain/TIGERweb_apps.html).

**The dashboard stop responding or the chart data disappeared!** This is likely caused by an Out of Memory Error and the application must be restarted.

**How do I request a feature or report a bug?** Create an [Issue](https://github.com/rapidsai/plotly-dash-rapids-census-demo/issues) and we will get to it asap.

## Acknowledgments and Data Sources

- 2020 Population Census and 2010 Population Census to compute Migration Dataset, used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org/) ( not for redistribution ).
- Base map layer provided by [Mapbox](https://www.mapbox.com/).
- Dashboard developed with [Plotly Dash](https://plotly.com/dash/).
- Geospatial point rendering developed with [Datashader](https://datashader.org/).
- GPU toggle accelerated with [RAPIDS cudf](https://rapids.ai/) and [cupy](https://cupy.chainer.org/), CPU toggle with [pandas](https://pandas.pydata.org/).
- For source code and data workflow, visit our [GitHub](https://github.com/rapidsai/plotly-dash-rapids-census-demo/tree/census-2020).
