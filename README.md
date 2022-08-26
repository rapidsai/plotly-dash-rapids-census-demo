# Plotly-Dash + RAPIDS | Census 2020 Visualization

![census_2020_readme_ss](https://user-images.githubusercontent.com/35873124/186991714-b8ec76f8-47d5-4663-b484-6b5666c5126b.png)

# Installation and Run Steps

## Base Layer Setup
The visualization uses a Mapbox base layer that requires an access token. Create one for free [here on mapbox](https://www.mapbox.com/help/define-access-token/). Go to the demo root directory's `plotly_demo` folder and create a token file named `.mapbox_token`. Copy your token contents into the file.

**NOTE:** Installation may fail without the token.


## Data 
There is 1 main dataset:

- 2020 Census for Total Population with Migration from 2010 Census

For more information on how the Census 2020 and 2010 Migration data was prepared to show individual points, refer to the `/data_prep_total_population` folder.
