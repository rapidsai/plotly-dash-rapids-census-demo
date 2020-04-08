# Plot.ly-Dash + RAPIDS | Census 2010 | Covid-19 Visualization

![screenshot](./RAPIDS-plotly%20Census-CV19%20v2.png)


# Installation and Run Steps

## Data 
There are 4 main datasets:

- 2010 Census for Population Density (~1.7GB) | download on first run
- 2018 ACS for County Population (~25KB) | downloaded on first run and included 
- 2019 HIFLD for Hospital Locations and Beds (~3.8MB) | downloaded on first run and included 
- Daily JHU CSSE Reports for Covid-19 Cases (~200KB per day) | downloaded every time app is started 

For more information on how the Census and ACS data was prepared to show individual points, refer to the [GitHub master Census Demo branch](https://github.com/rapidsai/plotly-dash-rapids-census-demo/tree/master).


## Base Layer Setup
The visualization uses a Mapbox base layer that requires an access token. Create one for free [here](https://www.mapbox.com/help/define-access-token/). In the root directory go to the `/plotly_demo` folder and create a token file named `.mapbox_token`. Copy your token contents into the file. **Note**: Installation will not succeed without a token. 

## Running the Visualization App

You can setup and run the visualization with the conda or docker commands below. Once the app is started, it will look for the datasets locally and if not found will **automatically download** them.


### Conda Env

```bash
# setup directory
cd plotly_demo

# create .mapbox_token

# setup conda environment ( root )
conda env create --name plotly_env --file environment.yml
source activate plotly_env

# run and access
python app-covid.py
```

### Docker

Verify the following arguments in the Dockerfile match your system:

1. CUDA_VERSION: Supported versions are `10.0, 10.1, 10.2`
2. LINUX_VERSION: Supported OS values are `ubuntu16.04, ubuntu18.04, centos7`

The most up to date OS and CUDA versions supported can be found here: [RAPIDS requirements](https://rapids.ai/start.html#req)

```bash
# setup directory
cd plotly_demo

# create .mapbox_token

# build ( root )
docker build -t plotly_demo .

# run and access via: http://localhost:8050 / http://ip_address:8050 / http://0.0.0.0:8050
docker run --gpus all -d -p 8050:8050 plotly_demo
```

## Dependencies

- plotly=4.5
- cudf
- dash=1.8
- pandas=0.25.3
- cupy=7.1
- datashader=0.10
- dask-cuda=0.12.0
- dash-daq=0.3.2
- dash_html_components
- gunicorn=20.0
- requests=2.22.0+
- pyproj


## FAQ and Known Issues
**What hardware do I need to run this locally?**  To run you need an NVIDIA GPU with at least **16GB of GPU memory**, and a Linux OS with driver and CUDA versions as defined in the [RAPIDS requirements](https://rapids.ai/start.html#req).


**How do I interact with the map?** Zoom in and out with a mouse wheel. Click the "pan" tool icon in the upper right corner of the map to pan. Click the "home" tool icon to reset the zoom level. 


**How are the population, hospital beds, and case counts filtered?** Click the "box select" tool icon in the upper right corner of the map, then click and drag an area to select. Click "clear selection" button to undo a selection. **Note**: because COVID-19 cases are only reported at the county level, we are placing the count bubble at the center of a county boundary. If a bubble is not part of the box selection, it will not be counted.


**Why is the population data from 2010?** Only decade census data is recorded on a block level for the full population. For more details on census boundaries refer to the [TIGERweb app](https://tigerweb.geo.census.gov/tigerwebmain/TIGERweb_apps.html). 


**How did you get individual point locations?** The population density points are randomly placed within a census block and associated to match distribution counts at a census block level. As such, they are not actual individuals, only a statistical representation of one.


**How did you get hospital beds and locations?** This is sourced from the HIFLD data noted below. Some hospital bed counts are unknown and this does not include emergency field hospitals or centers recently activated. 


**How are the bubbles sized?** The bubbles are set to a min size and a max size to aid in legibility, and as such are not a direct representation of the data. Hover over for exact counts. 


**How are state COVID-19 counts calculated?** State counts are aggregated from the county level. However, as some reports have no county associated with them, the visible county count may not match the state count. 


**Why can't I see a data layer I just toggled?** Sometimes the order of the toggle will put a layer under one another, interacting with the map should reset it.


**The dashboard stop responding or the chart data disappeared!** Try using the 'clear selections' button. If that does not work, refresh the page, or in the worst case restart the application. 


**How do I request a feature or report a bug?** Create an [Issue](https://github.com/rapidsai/plotly-dash-rapids-census-demo/issues), or ping on on this [Covid-19 Slack Channel](https://join.slack.com/t/rapids-goai/shared_invite/zt-2qmkjvzl-K3rVHb1rZYuFeczoR9e4EA) and we will get to it asap. 


## Acknowledgments and Data Sources

- 2010 Population Census and 2018 ACS data used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](www.nhgis.org) ( not for redistribution )
- Hospital data is from [HIFLD](https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals) (10.7.2019) and does not contain emergency field hospitals
- COVID-19 data is from the [Johns Hopkins University](https://coronavirus.jhu.edu/) data on [GitHub](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports) (updated daily)
- Base map layer provided by [mapbox](https://www.mapbox.com/)
- Dashboard developed with Plot.ly [Dash](https://dash.plotly.com/)
- Geospatial point rendering developed with [Datashader](https://datashader.org/)
- GPU accelerated with [RAPIDS](https://rapids.ai/) [cudf](https://github.com/rapidsai/cudf) and [cupy](https://cupy.chainer.org/) libraries
- For more information reach out with this [Covid-19 Slack Channel](https://join.slack.com/t/rapids-goai/shared_invite/zt-2qmkjvzl-K3rVHb1rZYuFeczoR9e4EA)
- For source code visit our [GitHub](https://github.com/rapidsai/plotly-dash-rapids-census-demo)
