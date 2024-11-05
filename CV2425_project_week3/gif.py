import os
from PIL import Image

case = 'vinijr'
pattern = 'checkboard' #checkboard or circle

output_file = f'./results/{case}_{pattern}_2.gif'

path = f'./results/{case}/{pattern}'


images_files = sorted(os.listdir(path))

images_list = []
for img in images_files:
    images_list.append(Image.open(f'{path}/{img}'))


jump = round(len(images_list)/250)
images_list = images_list[::jump]

time = 10

images_list[0].save(
    output_file,
    save_all=True,
    append_images=images_list[1:],
    optimize=True,
    duration= time*1000/len(images_list),  
    loop=0 #looping gif
)