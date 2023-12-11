import torch
from PIL import Image
from mylavis.models import my_load_model_and_preprocess

# load sample image
raw_image = Image.open("figs/happy.png").convert("RGB")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set max output length
max_len = 200 
# loads AU/Emot/Exp-BLIP model
# this also loads the associated image processors
checkpoint_path = './exp_blip_vitg_opt6.7b_trimmed.pth'
model, vis_processors, _ = my_load_model_and_preprocess(name="blip2_opt",
                model_type="caption_coco_opt6.7b", dict_path = dict_path, is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" 
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# generate caption
print('[1 caption]:',model.generate({"image": image},max_length=max_len))

# use nucleus sampling for diverse outputs 
print('[3 captions]:',model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3,max_length=max_len))
