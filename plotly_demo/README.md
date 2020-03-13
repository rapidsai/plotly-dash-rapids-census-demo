# Setup 


```bash
git clone https://github.com/AjayThorve/plotly_census_2010.git
cd plotly_census_2010
mkdir data
cd data
#download data from source
cd ../

conda create -n test_environment

conda update -n test_environment --file environment.yml
 
```


# Run

```bash
python app.py
```