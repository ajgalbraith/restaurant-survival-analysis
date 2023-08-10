import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install stuff
if __name__ == '__main__':
    install("dask")
    install("azureml-fsspec")
    # install("dask")


# only consider business IDs with over 500 reviews


# preprocess reviews

