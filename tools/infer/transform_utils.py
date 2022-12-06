import os
from PIL import  Image
import numpy as np
from torchvision import transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import pandas as pd 
import json
from tqdm import tqdm





def blur_transform(high):
    if high:
        return T.RandomApply([T.GaussianBlur(15, sigma=(1, 4))], p=0.3)
    else:
        return  T.RandomApply([T.GaussianBlur(11, sigma=(0.1, 2.0))], p=0.3)




def create_render_transform(high_blur,affine_degrees,translate,scale,contrast,brightness,saturation,hue,p_noise):
    """ This has language based operations - harmonise later"""
    return T.Compose([
        T.ToTensor(),
        T.RandomApply([T.RandomAffine(degrees=affine_degrees, translate=translate, fill=1)], p=0.7),
        T.RandomApply([T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)], p=0.5),
        T.ToPILImage(),
        lambda x: Image.fromarray(A.GaussNoise(var_limit=(10.0, 150.0), mean=0, p=p_noise)(image=np.array(x))["image"]),
        blur_transform(high_blur),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        T.ToPILImage()
    ])


###Tuples - scale,translate, all params are between 0,1 , translate has to have second number greater than 1st
###Randomly draw params and save them to a dict
def get_render_transform_params():
    scale = (1,1)
      ##Two copies of the same number randomly drawn
    translate_num = (random.uniform(0,0))
    translate = (translate_num,translate_num)
    affine_degrees = random.uniform(0, 0.2)
    contrast = random.uniform(0.8, 1.2)
    brightness = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.8, 1.2)
    hue = random.uniform(0.1, 0.4)
    p_noise = random.uniform(0.0, 0.3)
    high_blur = random.choice([True, False])
    return {"high_blur": high_blur,"affine_degrees": affine_degrees,"scale": scale, "translate": translate, "contrast": contrast,
            "brightness": brightness, "saturation": saturation, "hue": hue, "p_noise": p_noise}


def get_render_transform():
    params = get_render_transform_params()
    return create_render_transform(**params),params


###Run as script

if __name__ == "__main__":
    ##Load image df

    lang_code="en"
    input_df="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/Multilang_renders/"+lang_code+"/city_names.csv"
    input_df=pd.read_csv(input_df,encoding='utf-8-sig')

    ##Protopying - take only first 1000 images
    # input_df=input_df[:400000]

    ## output dir
    output_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/Noisy_Multilang_renders/"+lang_code+"/"
        

    ##Create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    ##Create output df
    output_df=pd.DataFrame(columns=["image_path","params","ground_truth"])
    ##Generate noisy images and save to output dir
    param_list=[]
    for i in tqdm(range(len(input_df))):
        image_path=input_df.iloc[i]["image_path"]
        image_name=image_path.split("/")[-1]
        transform,params=get_render_transform()
        image=Image.open(image_path)
        image=transform(image)
        image.save(os.path.join(output_dir,image_name))
        params["image_name"]=image_name
        # print("Saved image {}".format(image_name))
        param_list.append(params)

        ##Add params to df
        output_df=output_df.append({"image_path":os.path.join(output_dir,image_name),"params":json.dumps(params),"ground_truth":input_df.iloc[i]["ground_truth"]},ignore_index=True)


    # with open(os.path.join(output_dir,"params.json"), "w") as f:
    #     json.dump(param_list, f, indent=4)

    ##Save df
    output_df.to_csv(os.path.join(output_dir,"noisy_city_names.csv"),index=False)






