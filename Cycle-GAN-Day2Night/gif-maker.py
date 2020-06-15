import glob
from PIL import Image

# filepaths
fp_in = "D:\\Projects\\Intel-Project-Final\\UNIT-master\\UNIT-master\\outputs\\day2night\\images\\gen_a2b_test*.jpg"
fp_out = "day2night.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=100, loop=0)
