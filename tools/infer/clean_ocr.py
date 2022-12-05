####Multilingual OCR script using PAddle OCR

from paddleocr import PaddleOCR, draw_ocr
import pandas as pd
import numpy as np

# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="ch",use_gpu=True,ocr_version="PP-OCRv3") # The model file will be downloaded automatically when executed for the first time
# img_path ='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/Noisy_Multilang_renders/zh/龟龙山.png'
# Recognition and detection can be performed separately through parameter control

input_df_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/Noisy_Multilang_renders/zh/noisy_city_names.csv"
input_df=pd.read_csv(input_df_path,encoding='utf-8-sig')
input_df=input_df.head(4)
image_file_list=input_df.image_path.tolist()
image_file_list="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/Noisy_images/"
result = ocr.ocr(image_file_list, det=False,cls=False,rec=True)  

# Print detection frame and recognition result
for line in result:
    print(line)
