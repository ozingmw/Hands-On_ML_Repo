import os
import tarfile
import urllib.request
class github:
    def download(url, path, filename=None):
        if filename is None: filename = path.split("\\")[-1]
        if os.path.isfile(os.path.join(path, f'{filename}.csv')): return
        os.makedirs(path, exist_ok=True)
        tgz_path = os.path.join(path, f'{filename}.tgz')
        urllib.request.urlretrieve(url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=path)
        housing_tgz.close()