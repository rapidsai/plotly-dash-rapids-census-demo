#activating the conda environment
source activate rapids

cd /rapids/covid_demo

python download_datasets.py

gunicorn 'app-covid:server()'