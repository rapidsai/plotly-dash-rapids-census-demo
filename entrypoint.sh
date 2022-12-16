#activating the conda environment
source activate plotly_env

cd /rapids/plotly_census_demo/plotly_demo

if [ "$@" = "dask_app" ]; then
    python dask_app.py
else
    python app.py
fi

