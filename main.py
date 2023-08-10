import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install stuff
if __name__ == '__main__':
    install("dask")
    install("azureml-fsspec")
    # install("dask")

import dask.dataframe as dd

yelp_restaurants = dd.read_json("azureml://subscriptions/c9cf84fd-1186-4987-9964-23d999730aa0/resourcegroups/arthurjamesg-rg/workspaces/jgalb/datastores/workspaceblobstore/paths/UI/2023-08-09_184111_UTC/yelp_academic_dataset_business.json", lines=True)
yelp_restaurants.to_csv("./data/yelp-restaurants.csv")

yelp_reviews = dd.read_json(
    "azureml://subscriptions/c9cf84fd-1186-4987-9964-23d999730aa0/resourcegroups/arthurjamesg-rg/workspaces/jgalb/datastores/workspaceblobstore/paths/UI/2023-08-07_195630_UTC/yelp_academic_dataset_review.json",
     lines=True,
     nrows=10000
     )
yelp_reviews.to_csv("./data/yelp-reviews.csv")


google_reviews = dd.read_csv(
    "azureml://subscriptions/c9cf84fd-1186-4987-9964-23d999730aa0/resourcegroups/arthurjamesg-rg/workspaces/jgalb/datastores/workspaceblobstore/paths/UI/2023-08-09_180854_UTC/rating-New_York.csv.gz", 
    compression='gzip',
    blocksize=None
    )
google_reviews.to_csv("./data/google-reviews.csv")

google_restaurants = dd.read_json(
    "azureml://subscriptions/c9cf84fd-1186-4987-9964-23d999730aa0/resourcegroups/arthurjamesg-rg/workspaces/jgalb/datastores/workspaceblobstore/paths/UI/2023-08-09_183714_UTC/meta-New_York.json.gz",
     compression='gzip',
      blocksize=None,
      lines=True,
     nrows=1000)
google_restaurants.to_csv("./data/google-restaurants.csv")


# EDA