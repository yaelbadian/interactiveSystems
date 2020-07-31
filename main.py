# import gzip
# import json
# import tarfile

# with gzip.GzipFile('aotm2011_playlists.json.gz', 'r') as fin:
#     json_bytes = fin.read()

# json_str = json_bytes.decode('utf-8')
# data = json.loads(json_str)
# print(data)


import hdf5_getters
import numpy as np
import os
import re
import pandas as pd
from time import time
      
def hdf5_to_features(file_name):
    """
    Receives path to HDF5 file, returns 2 lists of identification for the song
    as well as the features for the algorithm.

    Parameters
    ----------
    file_name : str
        Absolute path to the HDF5 file.

    Returns
    -------
    list1 : list
        List consisting of ID, song title and artist name.

    list2 : list
        34 features to represent the song.
    """

    with hdf5_getters.open_h5_file_read(file_name) as reader:

        # ID
        ID = hdf5_getters.get_song_id(reader).decode()
        title = hdf5_getters.get_title(reader).decode()
        artist = hdf5_getters.get_artist_name(reader).decode()

        # Features 1-4
        beat_starts = hdf5_getters.get_beats_start(reader)
        beat_durations = np.diff(beat_starts, axis=0)
        # try:
        tempo_10, tempo_90 = np.quantile(beat_durations, [0.1, 0.9])
        # except:
        #     print(beat_durations)
        #     exit()
        temp_var = np.var(beat_durations)
        temp_mean = np.mean(beat_durations)

        # Features 5-8
        segment_loudness = hdf5_getters.get_segments_loudness_max(reader)
        loud_10, loud_90 = np.quantile(segment_loudness, [0.1, 0.9])
        loud_var = np.var(segment_loudness)
        loud_mean = np.mean(segment_loudness)

        # Features 9-21
        pitch_dominance = hdf5_getters.get_segments_pitches(reader)
        pitch_means = pitch_dominance.mean(axis=0)
        pitch_var = pitch_means.var()

        # Features 22-34
        timbre = hdf5_getters.get_segments_timbre(reader)
        timbre_means = timbre.mean(axis=0)
        timbre_var = timbre_means.var()

    return [ID, title, artist], [tempo_10, tempo_90, temp_var, temp_mean, loud_10, loud_90, loud_var, loud_mean] + list(pitch_means) + [pitch_var] + list(timbre_means) + [timbre_var]


def MSD_to_csv(path_to_MSD):
    failed = 0
    df = []
    start = time()
    files_done = 0
    for root, _, files in os.walk(path_to_MSD):
        for f in files:
            try:
                ID, features = hdf5_to_features(re.escape(root) + "\\" + f)
            except:
                failed += 1
                continue
            df.append(ID + features)
            files_done += 1
            if files_done % 100 == 0:
                print(f"Finished {files_done} files in {round((time() - start) / 60, 2)} minutes.")

    print("Failed", failed)
    df = pd.DataFrame(df,
     columns=["ID", "title", "artist", "tempo_10", "tempo_90", "tempo_var", "tempo_mean", "loud_10",
      "loud_90", "loud_var", "loud_mean", "pitch1", "pitch2", "pitch3", "pitch4",
       "pitch5", "pitch6", "pitch7", "pitch8", "pitch9", "pitch10", "pitch11", "pitch12",
        "pitch_var", "timbre1", "timbre2", "timbre3", "timbre4", "timbre5", "timbre6", "timbre7",
         "timbre8", "timbre9", "timbre10", "timbre11", "timbre12", "timbre_var"])
    df.to_csv("data.csv")
            



with hdf5_getters.open_h5_file_read(r"D:\data\MillionSongSubset\data\A\A\A\TRAAAAW128F429D538.h5") as reader:

    # print(hdf5_getters.get_title(reader))
    # print(hdf5_getters.get_artist_name(reader))
    print(hdf5_getters.get_segments_timbre(reader).shape)

MSD_to_csv(r"D:\data\MillionSongSubset\data")
df = pd.read_csv("data.csv")
