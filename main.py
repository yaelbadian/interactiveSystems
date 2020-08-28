# import gzip
# import json
# import tarfile
import pandas as pd
from DJ_MC import DJ_MC
from create_data import load_playlists, create_playlists
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

# with gzip.GzipFile('aotm2011_playlists.json.gz', 'r') as fin:
#     json_bytes = fin.read()

# json_str = json_bytes.decode('utf-8')
# data = json.loads(json_str)
# print(data)

# create_playlists()
data = pd.read_csv('data.csv', index_col=0).set_index("ID")
playlists = load_playlists()
# print(len(playlists))
# print(np.mean([len(x) for x in playlists]))
model = DJ_MC(data.drop(["title", "artist"], axis=1), playlists[0], 2, data[["title", "artist"]])
model.algorithm_5()

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)

with open("model_T=3000.pkl", 'rb') as f:
    model = pickle.load(f)
print((np.array([len(x) for x in playlists]) >= 3).sum())
print("random")
random_playlist, random_reward = model.random_predict()
print(random_reward)
print("greedy")
greedy_playlist, greedy_reward = model.greedy_predict()
print(greedy_reward)
print("DJ_MC")
DJ_MC_playlist, DJ_MC_reward = model.DJ_MC_predict()
print(DJ_MC_reward)

plt.plot(range(21), random_reward)
plt.plot(range(21), greedy_reward)
plt.plot(range(21), DJ_MC_reward)
plt.legend(["Random", "Greedy", "DJ-MC"])
plt.xlabel("Playlist Length")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Of Playlist Up To Index")
plt.show()