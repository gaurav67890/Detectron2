import json

mode='train.json'

new_data={}
with open('thin/'+mode) as f:
    thin_data=json.load(f)
with open('big/'+mode) as f:
    big_data=json.load(f)

thin_data_images={}
big_data_images={}

for i in range(len(thin_data['images'])):
    thin_data_images[thin_data['images'][i]['file_name']]=thin_data['images'][i]['id']
for i in range(len(big_data['images'])):
    big_data_images[big_data['images'][i]['id']]=big_data['images'][i]['file_name']

for i in range(len(big_data['images'])):
    big_data['images'][i]['id']=big_data['images'][i]['id']+10000

remove_image_id=[]
for i in range(len(big_data['annotations'])):
    big_data['annotations'][i]['id']=big_data['annotations'][i]['id']+10000
    big_data['annotations'][i]['category_id'] = 1
    if big_data['annotations'][i]['image_id'] in big_data_images.keys():
        file_name=big_data_images[big_data['annotations'][i]['image_id']]
        if file_name in thin_data_images.keys():
            image_id_new= thin_data_images[file_name]
            big_data['annotations'][i]['image_id']=image_id_new
            remove_image_id.append(big_data['annotations'][i]['image_id'])
        else:
            big_data['annotations'][i]['image_id'] = big_data['annotations'][i]['image_id'] + 10000
    else:
        big_data['annotations'][i]['image_id'] = big_data['annotations'][i]['image_id'] + 10000

for i in range(len(thin_data['images'])):
    big_data['images']=[x for x in thin_data['images'] if x['id'] not in remove_image_id]


big_data['categories'][0]['id']=1

new_data['info']=big_data['info']
new_data['licences']=big_data['licences']
new_data['images']=thin_data['images']+big_data['images']
new_data['annotations']=thin_data['annotations']+big_data['annotations']
new_data['categories']=thin_data['categories']+big_data['categories']

with open('new_merge_'+mode,'w') as f:
    json.dump(new_data,f,indent=4,ensure_ascii = False)


