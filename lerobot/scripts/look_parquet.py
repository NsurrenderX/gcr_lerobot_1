import pandas as pd

path = "/data_16T/lerobot_openx/furniture_bench_dataset_lerobot/merged.parquet"
path = "/data_16T/lerobot_openx/furniture_bench_dataset_lerobot/data/chunk-005/episode_005099.parquet"

df = pd.read_parquet(path, engine="pyarrow")
print(df.head())
for index, row in df.iterrows():
    print(index, row)
    