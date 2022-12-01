##Generate report with result, ground trth and images
##Write a pdf with the images and the results and ground truth


import pandas as pd
import matplotlib.pyplot as plt 
import os
import matplotlib.font_manager as fm
fprop = fm.FontProperties(fname='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/all_fonts/BabelStoneHan.ttf')

###Run as script

if __name__ == "__main__":
    ##Load error df
    error_df="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/ocr_hg_errors.csv"
    error_df=pd.read_csv(error_df,encoding='utf-8-sig')

    image_output_dir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/homoglyphic_errors/"

    ##If output dir does not exist, create it
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)


    for i in range(len(error_df)):
        ##Plot the image and add result and ground truth as caption
        image_path=error_df.iloc[i]["image_path"]
        image_name=image_path.split("/")[-1]
        result=error_df.iloc[i]["result"]
        ground_truth=error_df.iloc[i]["ground_truth"]
        img=plt.imread(image_path)
        
        ##Plot image - high res plot!
        plt.figure()
        plt.imshow(img)
        plt.title("Result: {} Ground truth: {}".format(result,ground_truth),fontproperties=fprop,fontsize=20)
        plt.savefig(os.path.join(image_output_dir,image_name))
        plt.close()
        print("Saved image {}".format(image_name))





