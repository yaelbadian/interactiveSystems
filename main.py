# import gzip
# import json
# import tarfile
import pandas as pd

# with gzip.GzipFile('aotm2011_playlists.json.gz', 'r') as fin:
#     json_bytes = fin.read()

# json_str = json_bytes.decode('utf-8')
# data = json.loads(json_str)
# print(data)


data = pd.read_csv('data.csv')
