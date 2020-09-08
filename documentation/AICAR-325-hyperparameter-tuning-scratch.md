# Hyper-parameter tuning in GCP

### Note: Just incase you are at master branch, switch to feat/AICAR-325-hyperparameter-tuning-scratch branch. For GCP training, all the required code is inside feat/AICAR-325-hyperparameter-tuning-scratch branch, not inside master.

### 1. Install google cloud and login https://cloud.google.com/sdk/docs
```
gcloud auth login
```

### 2. If the above installaton shows error- gcloud not found, then run below commond
```
curl https://sdk.cloud.google.com | bash
```

### 3. Also it is important to configure your docker with gcloud, otherwise it will show permission error
```
gcloud auth configure-docker
```

### 4. Then we will create image, but before that we will export some values

```
#Make a new bucket for your project, and assign that name to BUCKET_NAME
export BUCKET_NAME=hptuning2
#Project id will be ai-project-231602
export PROJECT_ID=ai-project-231602
export JOB_DIR=gs://$BUCKET_NAME/hp_job_dir
export IMAGE_REPO_NAME=scartch_tuning_container
export IMAGE_TAG=scratch_tuning
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
export REGION=us-west1
export JOB_NAME=hp_tuning_scratch_container_job_$(date +%Y%m%d_%H%M%S)

```

### 5. git clone this code, in order to use the Dockerfile and hyper_tuning.config file
#feat/AICAR-325-hyperparameter-tuning-scratch branch
```
git clone https://github.com/gaurav67890/Detectron2 detectron2_repo -b feat/AICAR-325-hyperparameter-tuning-scratch
```
### 6. Take the credentials.json file and copy it inside above branch 
```
cd Detectron2
cp /path/to/credentials.json ./
```

### 7. Create the image using the Dockerfile from the above code.
```
docker build --no-cache --build-arg USER_ID=$UID -f Dockerfile -t $IMAGE_URI ./

Note: Make changes inside the last line of Dockerfile(ENTRYPOINT) according to your requirements.

1.trainer-main-gpu-single.py is for single gpu
2.trainer-main-gpu.py is for multiple gpus
3. Also you can replace dent with crack, scratch,broken etc.
```

### 8. Get the image id for the above image, and use it to push the docker image to gcp
```
docker tag image_ID $IMAGE_URI
docker push $IMAGE_URI

```

### 9. We are almost done, now we just need to run this below command and start the gcp training
```
gcloud ai-platform jobs submit training $JOB_NAME --job-dir $JOB_DIR --region $REGION --master-image-uri $IMAGE_URI --config hyper_tuning.yaml
```
### 10. See the status of job using
```
gcloud ai-platform jobs describe hp_tuning_scratch_container_job_20200819_204006

#this will generate the url for the job status and log which you can click. 
```
### 11. To calculate the DICE and MAP locally, make use of custom_test_net_MAP.py and custome_test_net_DICE.py


### 12. All the path related parameters are present inside params.yaml

