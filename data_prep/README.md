# Census 2010 dataset generation

The census_data we prepared for this demo is basically a compilation and estimation of 3 datasets:
1. Census 2010 sf1 
2. Acs_2006_2010
3. Shape files(tiger) from 2010

Steps we follow to compile and estimate:
 - Take the block wise population from 2010 sf1 dataset
 - Assign random lat-longs (accurate within a block) using polygons from shape files
 - Results in a dataset with person_id, lat, long, Block_id
 - Take Block-group level estimates of age, sex, income and education from acs dataset, and distribute them for each block 
 
 > Note: IMPORTANT: We assume block-group level estimates for individual blocks, which may not result in the most accurate estimates of age, income, education and sex distributions. This is intended for demonstration of the viz libraries only


The final dataset can be downloaded [here](https://s3.us-east-2.amazonaws.com/rapidsai-data/viz-data/census_data.parquet.tar.gz)


However, if you want to recreate the dataset yourself, please follow the following steps:

## Step 1: Download initial data

Go to nhgis data finder and use the following filters

### 1. Total Population (2010 Block)
    Filters: 

    Geographic Levels: BLOCK
    YEARS: 2010

    Final File: P1. Total Population |	Total population |		2010_SF1a |	Spatial

### 2. Shape files
    Filters:

    Geographic Levels: BLOCK
    YEARS: 2010

    Final File: Select all 52 GIS files

### 3. ACS 2010 distributions
    Filters:

    Geographic Levels: BLCK_GRP
    YEARS: 2006-2010
    TOPICS: Age or Sex or Educational Attainment or Class of Worker or Personal Income

    Select the following files:
    - B01001. Sex by Age
    - B15002. Sex by Educational Attainment for the Population 25 Years and Over
    - B20001. Sex by Earnings in the Past 12 Months (in 2010 Inflation-Adjusted Dollars) for the Population 16 Years and Over with Earnings in the Past 12 Months
    - B24080. Sex by Class of Worker for the Civilian Employed Population 16 Years and Over

    Year-DATASET: 2006_2010_ACS5a


## Step 2: Execute scripts in the following order

1. data_set_prep_concat_states
    - gen_points_script.py (takes around 24 hours to finish execution - assign each of 300+M people a lat and long as per shape files)
    - concat_states.ipynb
2. add_gender_to_dataset
3. add_age_to_dataset
4. add_education_to_dataset
5. add_income_to_dataset
6. add_cow_to_dataset (cow -> class of workers)