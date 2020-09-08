# Hyper-parameter tuning in GCP for Augmented dent dataset



### 1. All the steps from building docker and running gcp will remain same from feat/AICAR-325-hyperparameter-tuning-scratch branch

### 2. Addition of mode has been done, where we select ORIGINAL or AUGMENTED for the type of dataset we want to use for training. All the parameters related to path are presents inside params.yaml 

### 3. Also it is important to note that for MODE= AUGMENTED we cannot use any damage other than dent for now, since we are specifically downloading the dent dataset from google storage.




