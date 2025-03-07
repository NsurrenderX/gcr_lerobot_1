import pandas as pd

path = "/data_16T/lerobot_openx/furniture_bench_dataset_lerobot/merged.parquet"

df = pd.read_parquet(path, engine="pyarrow")
print(df.head())
for index, row in df.iterrows():
    print(index, row)
    