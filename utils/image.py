from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    img = img.convert("RGB")
    img = img.resize((224, 224), resample=Image.BICUBIC)
    return img
