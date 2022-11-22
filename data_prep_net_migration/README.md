# Net Migration dataset generation

## Order of execution

1. gen_table_with_race_migration
2. gen_race_mig_points
3. compute_race
4. assign_race

## Mappings:

### Block Net

<b>1</b>: Inward Migration</br>
<b>0</b>: Stationary</br>
<b>-1</b>: Outward Migration</br>

### Block diff

Integer

### Race

<b>0</b>: All</br>
<b>1</b>: White</br>
<b>2</b>: African American</br>
<b>3</b>: American Indian</br>
<b>4</b>: Asian alone</br>
<b>5</b>: Native Hawaiian</br>
<b>6</b>: Other Race alone</br>
<b>7</b>: Two or More</br>

### County

Mappings for counties can be found in `id2county.pkl` file from root directory.

### Final Dataset

You can download the final net miragtion dataset [here](https://rapidsai-data.s3.us-east-2.amazonaws.com/viz-data/net_migration_dataset.parquet)
