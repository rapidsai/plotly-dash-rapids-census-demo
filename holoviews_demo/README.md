# Panel + Holoviews + RAPIDS | Census 2020 Race Migration Visualization

![hvr1](https://user-images.githubusercontent.com/35873124/189291984-d95ddf27-9ec8-452a-b596-05398ce47969.png)

## Charts

1. Map chart shows the total migration points for chosen view and selected area
2. Top counties bar show the counties with most migration for chosen view and selected area
3. Net Race migration bar shows total inward and outward migration for chosen view and selected area
4. Population Distribution shows distribution of migration across blocks for chosen view and selected area

Cross-filtering is enabled to link all the four charts using box-select tool

## Race Views

The demo consists of eight views ( seven race views + one all-race view)

Options - All, White alone, African American alone, American Indian alone, Asian alone, Native Hawaiian alone, Other Race alone, Two or More races.

#### Snapshot examples

1. White race

![white](https://user-images.githubusercontent.com/35873124/189290231-4f573dba-6357-4f0a-89cd-14260fa35d0b.png)

2. Asian race

![asian](https://user-images.githubusercontent.com/35873124/189290237-bdece601-4237-436a-a90f-039f42790b9c.png)

3. African american race

![africanamerican](https://user-images.githubusercontent.com/35873124/189290258-27aa8b71-cdfc-443b-99d9-260b2bbcd342.png)

## Colormaps

User can select from select colormaps

Options - 'kbc', 'fire', 'bgy', 'bgyw', 'bmy', 'gray'.

## Limit

User can use slider to select how many top counties to show, from 5 to 50 at intervals of 5

# Installation and Run Steps

## Data

There is 1 main dataset:

- Net Migration Dataset ; Consists of Race Migration computed using Census 2020 and Census 2010 block data

For more information on how the Net Migration Dataset was prepared to show individual points, refer to the `/data_prep_net_migration` folder.

You can download the final net miragtion dataset [here](https://data.rapids.ai/viz-data/net_migration_dataset.parquet)

### Conda Env

Verify the following arguments in the `environment.yml` match your system(easy way to check `nvidia-smi`):

cudatoolkit: Version used is `11.5`

```bash
# setup conda environment
conda env create --name holoviews_env --file environment.yml
source activate holoviews_env

# run and access
cd holoviews_demo
jupyter lab
run `census_net_migration_demo.ipynb` notebook
```

## Dependencies

- python=3.9
- cudatoolkit=11.5
- rapids=22.08
- plotly=5.10.0
- jupyterlab=3.4.3

## FAQ and Known Issues

**What hardware do I need to run this locally?** To run you need an NVIDIA GPU with at least 24GB of memory, at least 32GB of system memory, and a Linux OS as defined in the [RAPIDS requirements](https://rapids.ai/start.html#req).

**How did you compute migration?** Migration was computed by comparing the block level population for census 2010 and 2020

**How did you compare population having block level boundary changes?** [Relationship Files](https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.html#t10t20) provides the 2010 Census Tabulation Block to 2020 Census Tabulation Block Relationship Files. Block relationships may be one-to-one, many-to-one, one-to-many, or many-to-many. Population count was computed in proportion to take into account the division and collation of blocks across 2010 and 2020.

**How did you determine race migration?** We took difference of race counts for census 2020 and census 2010. Individuals were randomly assigned race within a block so that they accurately add up at the block level.

**How did you get individual point locations?** The population density points are randomly placed within a census block and associated to match distribution counts at a census block level.

**How are the population and distributions filtered?** Use the box select tool icon for the map or click and drag for the bar charts.

**Why is the population data from 2010 and 2020?** Only census data is recorded on a block level, which provides the highest resolution population distributions available. For more details on census boundaries refer to the [TIGERweb app](https://tigerweb.geo.census.gov/tigerwebmain/TIGERweb_apps.html).

**The dashboard stop responding or the chart data disappeared!** This is likely caused by an Out of Memory Error and the application must be restarted.

**How do I request a feature or report a bug?** Create an [Issue](https://github.com/rapidsai/plotly-dash-rapids-census-demo/issues) and we will get to it asap.

## Acknowledgments and Data Sources

- 2020 Population Census and 2010 Population Census to compute Net Migration Dataset, used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org/) ( not for redistribution ).
- Dashboard developed with [Panel](https://panel.holoviz.org/) and [Holoviews](https://holoviews.org/index.html)
- Geospatial point rendering developed with [Datashader](https://datashader.org/).
- GPU acceleration with [RAPIDS cudf](https://rapids.ai/) and [cupy](https://cupy.chainer.org/), CPU code with [pandas](https://pandas.pydata.org/).
- For source code and data workflow, visit our [GitHub](https://github.com/rapidsai/plotly-dash-rapids-census-demo/tree/census-2020).
