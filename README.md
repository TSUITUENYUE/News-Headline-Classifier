## Setup: 
```git clone https://github.com/TSUITUENYUE/News-Headline-Classifier.git``` 

```conda env create -f environment.yml``` 

```conda activate newsCLS```
## Data Preparation:
put url_only_data.csv under the data/fox_nbc directory

```python data_collection.py```

## Run code: 
```python main.py --mode train --conf ./confs/binarycls.conf --case <case_name> ```

