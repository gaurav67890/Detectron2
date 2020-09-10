import json

mode='train.json'

new_data={}
with open('thin_scratch/'+mode) as f:
    thin_data=json.load(f)
with open('big_scratch/'+mode) as f:
    big_data=json.load(f)


for i in range(len(big_data['images'])):
    big_data['images'][i]['id']=big_data['images'][i]['id']+10000
for i in range(len(big_data['annotations'])):
    big_data['annotations'][i]['image_id']=big_data['annotations'][i]['image_id']+10000
    big_data['annotations'][i]['id']=big_data['annotations'][i]['id']+10000
    big_data['annotations'][i]['category_id'] = 1

big_data['categories'][0]['id']=1

new_data['info']=big_data['info']
new_data['licences']=big_data['licences']
new_data['images']=thin_data['images']+big_data['images']
new_data['annotations']=thin_data['annotations']+big_data['annotations']
new_data['categories']=thin_data['categories']+big_data['categories']

with open('merged_scratch/merged_'+mode,'w') as f:
    json.dump(new_data,f,indent=4,ensure_ascii = False)


