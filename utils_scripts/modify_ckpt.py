from typing import OrderedDict
import torch 
import re
import os



ckpt_uncounatble_path = "/data/ybyang/out/trained_model/urbangiraffe/urbangiraffe/None/latest.pth"

ckpt_sky_path = ckpt_uncounatble_path 
ckpt_building_path = ckpt_uncounatble_path 
ckpt_car_path = '/data/ybyang/out/trained_model/urbangiraffe/urbangiraffe_car_pc/gan:1.0_l2:10.0_gan_obj:1.0_gan_reg:10.0_perceptual:2.0_NR32_semantcis:car_N:4_pixel:5000/38.pth'
ckpt_D_path = ckpt_uncounatble_path

ckpt_render_path = "/data/ybyang/out/trained_model/urbangiraffe/urbangiraffe/None"

ckpt_sky = torch.load(ckpt_sky_path, 'cpu')
ckpt_car= torch.load(ckpt_car_path, 'cpu')
ckpt_building= torch.load(ckpt_building_path, 'cpu')
ckpt_uncountable= torch.load(ckpt_uncounatble_path, 'cpu')
ckpt_D= torch.load(ckpt_D_path, 'cpu')

sky_net, car_net, building_net, uncountable_net, D_net = ckpt_sky['net'], ckpt_car['net'],ckpt_building['net'], ckpt_uncountable['net'], ckpt_D['net']

ckpt_new = ckpt_uncountable.copy()
new_net = OrderedDict()


for k in uncountable_net:
    if re.search('generator', k) != None:
        new_net[k] = uncountable_net[k]
for k in car_net:
      if re.search('car', k) != None or re.search('obj', k) != None :
        new_net[k] = car_net[k]
# for k in building_net:
#       if re.search('building', k) != None or re.search('triplane', k) != None :
#         new_net[k] = building_net[k]
for k in sky_net:
    if re.search('sky', k) != None:
        new_net[k] = sky_net[k]

for k in D_net:
    if re.search('discriminator', k) != None:
        new_net[k] = D_net[k]


ckpt_new['net'] = new_net
if not os.path.exists(ckpt_render_path):
            os.makedirs(ckpt_render_path)
torch.save(ckpt_new, os.path.join(ckpt_render_path, 'latest.pth'))
