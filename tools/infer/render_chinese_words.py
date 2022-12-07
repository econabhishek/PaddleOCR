###generate chinese word renders

from numpy.lib.function_base import kaiser
# import torchvision.transforms as T
import numpy as np
import os
from tqdm import tqdm

from PIL import ImageOps, Image, ImageFont, ImageDraw
import os
from tqdm import tqdm

from fontTools.unicode import Unicode

import pandas as pd


def crops_from_text(text, font, font_size=256,random_size=True,random_scale_canvas=False):
    n=len(text)
    
    p = font_size // 25
    crops = []
    
    for c in text:
        img = Image.new('RGB', (font_size*4, font_size*4), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((font_size,font_size), c, (255, 255, 255), font=font, anchor='mm')
        bbox = img.getbbox()
        if bbox is None:
            n -= 1
            continue
        x0,y0,x1,y1 = bbox
        pbbox = (x0-p,y0-p,x1+p,y1+p)
        crop = ImageOps.invert(img.crop(pbbox))
        crops.append(crop)
    
    if random_scale_canvas:
        rand_scal_w = np.random.uniform(1.0, 1.4)
        rand_scal_h = np.random.uniform(1.1, 1.2)
        canvas_w = int(font_size*n*rand_scal_w)
        canvas_h = int(font_size*rand_scal_h)
    else:
        canvas_w = int(font_size*n*1.2)
        canvas_h = int(font_size*1.2)
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    
    coco_bboxes = []
    x = 0 # np.random.uniform(0, font_size) ## This is to randomize the starting point of the first character
    for i in range(n):
        if text[i] == "_":
            x += font_size
            continue
        if text[i] == ",":
            y = canvas_h - font_size // 3
        else:
            y = font_size // 15
        pcrop = crops[i]
        if random_size:
            pcrop = pcrop.resize((int(pcrop.size[0]*np.random.normal(1, 0.1)), 
                int(pcrop.size[1]*np.random.normal(1, 0.1))))
        w, h = pcrop.size
        rand_x = np.random.uniform(0, 0.25)
        rand_y = np.random.uniform(0.97, 1.03)
        x = int(w * rand_x) + int(x)
        y = int(y * rand_y)
        x= int(x)
        y=int(y)
        canvas.paste(pcrop, (x, y, x + w, y + h))
        coco_bboxes.append((x, y, w, h))
        x += w
    
    return coco_bboxes, canvas


def draw_word_from_text(text,font,font_size):
    """Draw a word given a text string"""
    n=len(text)
    img = Image.new('RGB', (font_size*n, font_size*2), (0,0,0))
    draw = ImageDraw.Draw(img)
    draw.text((0,0), text, (255, 255, 255), font=font,anchor='ms')

    # ##Get bb and crop image with that bb
    # img_copy=img.copy()
    # ##Invert a copy of the image
    # img_copy = ImageOps.invert(img_copy)

    # bbox = img.getbbox()
    # x0,y0,x1,y1 = bbox
    # # p = font_size // 25
    # # pbbox = (x0-p,y0-p,x1+p,y1+p)

    # ##Add some padding
    # p = font_size // 25
    # pbbox = (x0-p,y0-p,x1+p,y1+p)
    # crop = img.crop(pbbox)

    # crop=ImageOps.invert(crop)

    return img



def draw_word_from_text(text,font,font_size):
    """Draw a word given a text string"""
    n=len(text)
    img = Image.new('RGB', (font_size*n*4, font_size*n*4), (255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((font_size*4,font_size*4), text, (0, 0, 0), font=font,anchor='ms',align='center')

    ##Get bb and crop image with that bb
    img_copy=img.copy()
    ##Invert a copy of the image
    img_copy = ImageOps.invert(img_copy)

    bbox = img_copy.getbbox()
    x0,y0,x1,y1 = bbox
    # p = font_size // 25
    # pbbox = (x0-p,y0-p,x1+p,y1+p)

    ##Add some padding
    p = font_size // 25
    pbbox = (x0-p,y0-p,x1+p,y1+p)
    crop = img.crop(pbbox)


    return img





##Run as script
if __name__ == "__main__":
    ##Load the city names file from csv
    font_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/all_fonts/NotoSans-Regular.ttf"
    ###Import csv containing city names in china from wikipedia
    path_to_names = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multilang_gts/US_en_alt_names.csv"

    ##Image save path
    save_path = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/Multilang_renders/en/"

    ##Create save path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.read_csv(path_to_names)
    df = df.dropna()
    ##Get only the city names
    city_names = df['name'].to_numpy()

    ##Drop city names that have a "/" in them
    city_names = [x for x in city_names if "/" not in x]
    # city_names = [x.split(" ")[0] for x in city_names]
    # city_names = [x.split(",")[0] for x in city_names]
    # city_names = [x.split("-")[0] for x in city_names]
    # city_names = [x.split("/")[0] for x in city_names]

    ###Take unique city names
    city_names = np.unique(city_names)

    ##Take random sample of 400k city names
    city_names = np.random.choice(city_names, 10)
    font_size=133

    font = ImageFont.truetype(font_path, font_size)




    ##Iterate through the city names and generate images
    ##Empty result df
    result_df = pd.DataFrame(columns=['ground_truth','image_path','font_path'])
    for i in tqdm(range(len(city_names))):

        text = city_names[i]

       
        

        # coco_bboxes, canvas = crops_from_text(text, font, font_size=font_size)
        canvas=draw_word_from_text(text,font,font_size)
        # canvas.show()
        city_file_name=text.replace(" ","_")
        image_path=save_path+city_file_name+".png"
        
        # canvas.show()
        canvas.save(image_path)

        ##Add to result df
        result_df = result_df.append({'ground_truth':text,'image_path':image_path,'font_path':font_path},ignore_index=True)


    result_df.to_csv(save_path+'city_names.csv',index=False,encoding='utf-8-sig')

    