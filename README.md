# Plotly-Dash + RAPIDS | Census 2020 Visualization

![census_2020_readme_ss](https://user-images.githubusercontent.com/35873124/186991714-b8ec76f8-47d5-4663-b484-6b5666c5126b.png)

## Data-Selection Views
The demo consists of six views and all views are calculated at a block level
- Total Population view shows total Census 2020 population.
- Migrating In view shows net inward decennial migration.
- Stationary view shows population that were stationary.
- Migrating Out view shows net outward decennial migration.
- Net Migration view shows total decennial migration. Points are colored into three categories - migrating in, stationary, migrating out
- Population with Race shows total Census 2020 population colored into seven race categories - White alone, African American alone, American Indian alone, Asian alone, Native Hawaiian alone, Other Race alone, Two or More races.


# Installation and Run Steps

## Base Layer Setup
The visualization uses a Mapbox base layer that requires an access token. Create one for free [here on mapbox](https://www.mapbox.com/help/define-access-token/). Go to the demo root directory's `plotly_demo` folder and create a token file named `.mapbox_token`. Copy your token contents into the file.

**NOTE:** Installation may fail without the token.


## Data 
There is 1 main dataset:

- 2020 Census for Total Population with Migration from 2010 Census

For more information on how the Census 2020 and 2010 Migration data was prepared to show individual points, refer to the `/data_prep_total_population` folder.
