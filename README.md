# us-pilot-synthetic-data-gen
Scripts for generating synthetic data for the US pilot 


Prepare the environment and install dependencies 

```
git clone https://github.com/world-federation-of-advertisers/us-pilot-synthetic-data-gen
cd us-pilot-synthetic-data-gen
python3 -m venv ../synthetic-data-env
source ../synthetic-data-env/bin/activate
pip3 install -r requirements.txt 
```

Then run with providing
 1. EDP id. Either Digital1, Digital2 or Digital3
 2. A random seed

This command will generate a set of files in the form of `Digital1_row_*_fake_data.csv` in your directory: 

```
python3 main.py -r 4 -e Digital1
```

It should take ~30 mins to generate the whole dataset with 8 cores