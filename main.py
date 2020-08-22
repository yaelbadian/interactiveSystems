# import gzip
# import json
# import tarfile
import pandas as pd
from DJ_MC import DJ_MC
from create_data import load_playlists, create_playlists
import numpy as np

# with gzip.GzipFile('aotm2011_playlists.json.gz', 'r') as fin:
#     json_bytes = fin.read()

# json_str = json_bytes.decode('utf-8')
# data = json.loads(json_str)
# print(data)

create_playlists()
data = pd.read_csv('data.csv', index_col=0).set_index("ID")
playlists = load_playlists()
print(len(playlists))
print(np.mean([len(x) for x in playlists]))
model = DJ_MC(data.drop(["title", "artist"], axis=1), playlists[0], 20)
model.algorithm_5()
