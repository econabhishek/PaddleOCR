###PAddleocr Model Downloader

import os
import requests


def download_model(model_link,save_dir):
    downloaded_name=model_link.split("/")[-1]
    if os.path.exists(save_dir+downloaded_name):
        print("Model already exists")
        return
    else:
        r=requests.get(model_link)
        with open(save_dir+downloaded_name,"wb") as f:
            f.write(r.content)
        print("Model downloaded")

if __name__=="__main__":
    model_link="https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar"
    save_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/configs/"
    download_model(model_link,save_dir)

    ##Extract file and remove tar file
    os.system("cd "+save_dir+" && tar -xf "+ model_link.split("/")[-1])

    ##Delete tar file
    os.system("rm "+save_dir+model_link.split("/")[-1])
