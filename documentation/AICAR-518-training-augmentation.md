# Hyper-parameter tuning in GCP for Augmented dent dataset



### 1. All the steps from building docker and running gcp will remain same from feat/AICAR-325-hyperparameter-tuning-scratch branch


### 2. Changes are made inside the Dockerfile where we used trainer-main-gpu-aug.py python file and instead of scraping split_damages dataset, we are using dent_aug dataset from google storage.

### 3. Also it is important to note that in this branch we cannot use any damage other than dent, since we are specifically downloading the dent dataset from google storage.




