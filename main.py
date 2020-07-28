import gzip
import json
import tarfile

with gzip.GzipFile('aotm2011_playlists.json.gz', 'r') as fin:
    json_bytes = fin.read()

json_str = json_bytes.decode('utf-8')
data = json.loads(json_str)
print(data)
