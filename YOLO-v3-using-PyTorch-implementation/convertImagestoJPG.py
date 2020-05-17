from PIL import Image
import os
if __name__ == "__main__":
    folder = input("Enter input folder")
    output = input('output folder')
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        if img is not None:
            out = img.convert("RGB")
            out.save(output+filename.split('.')[0]+'.jpg', quality=90)
            print('done '+filename.split('.')[0])