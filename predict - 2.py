import os
from PIL import Image
from deeplab import DeeplabV3
from tqdm import tqdm

if __name__ == "__main__":
    deeplab = DeeplabV3()
    mode = "dir_predict"

    img_base_path = r'D:\论文2\Improved-deeplabv3-plus-sp - ca2副本\VOCdevkit'
    dir_save_path = r"D:\论文2\Improved-deeplabv3-plus-sp - ca2副本\img"
    base_path = r"D:\论文2\Improved-deeplabv3-plus-sp - ca2副本\VOCdevkit\VOC2007\ImageSets\Segmentation"
    txt_file_path = os.path.join(base_path, "test.txt") 

    with open(txt_file_path, 'r') as file:
        img_names = [line.strip() + ".jpg" for line in file]   

    if mode == "dir_predict":
        os.makedirs(dir_save_path, exist_ok=True)  # 确保保存路径存在
        for img_name in tqdm(img_names):
            image_path = os.path.join(img_base_path, "VOC2007", "JPEGImages", img_name)
            try:
                if os.path.exists(image_path):
                    print(f"Processing {image_path}")
                    image = Image.open(image_path)
                    r_image = deeplab.detect_image(image)
                    
                    # 将保存路径扩展名改为 .png
                    save_path = os.path.join(dir_save_path, os.path.splitext(img_name)[0] + ".png")
                    
                    # 检查 r_image 是否有效
                    if r_image is not None and isinstance(r_image, Image.Image):
                        r_image.save(save_path, format="PNG")  # 指定保存格式为 PNG
                        print(f"Saved: {save_path}")
                    else:
                        print(f"Failed to process image: {img_name}")
                else:
                    print(f"Image not found: {image_path}")
            except Exception as e:
                print(f"Could not process {img_name}: {e}")

    print(f"All predictions saved to {dir_save_path}")

