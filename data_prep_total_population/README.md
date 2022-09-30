# Total population dataset generation


## Order of execution

1. map_blocks_and_calc_population
2. gen_table_with_migration
3. gen_total_population_points_script
4. add_race_net_county_to_population


## Mappings:


### Net

<b>1</b>:     Inward Migration</br>
<b>0</b>:     Stationary</br>
<b>-1</b>:    Outward Migration</br>

### Race

<b>0</b>:     All</br>
<b>1</b>:     White</br>
<b>2</b>:    African American</br>
<b>3</b>:     American Indian</br>
<b>4</b>:     Asian alone</br>
<b>5</b>:     Native Hawaiian</br>
<b>6</b>:     Other Race alone</br>
<b>7</b>:     Two or More</br>

### County

Mappings for counties can be found in `id2county.pkl` file from root directory.

