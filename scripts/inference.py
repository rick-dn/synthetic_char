import torch
import os
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from ip_adapter import IPAdapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
import cv2
from insightface.app import FaceAnalysis
import random

# Change the working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# set refernce dirs and check
LORA_WEIGHTS="../data/sbsbbx_beachybeach_lora_model/"
REF_IMGS_DIR = "../data/ref_imgs/"
IP_ADAPTER_CKPT = "../IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin"
OUT_DIR = "../outputs/"
assert os.path.isdir(LORA_WEIGHTS)
assert os.path.isdir(REF_IMGS_DIR)
assert os.path.isfile(IP_ADAPTER_CKPT)

# ✅ Load base SD 1.5 model with LoRA weights
print("# ✅ Load base SD 1.5 model with LoRA weights")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to("cuda")
pipe.safety_checker = None

# ✅ Load LoRA weights trained on sbsbbx beachybeach identity
print("✅ Load LoRA weights trained on sbsbbx beachybeach identity")
pipe.load_lora_weights(LORA_WEIGHTS)
# Optional: fuse LoRA into base weights (permanently applies it)
# pipe.fuse_lora()


# ✅ Select a random reference image and extract embeddings
print("✅ Select a random reference image and extract embeddings")
valid_exts = [".png", ".jpg", ".jpeg"]
ref_image_paths = [
    os.path.join(REF_IMGS_DIR, f)
    for f in os.listdir(REF_IMGS_DIR)
    if os.path.splitext(f)[1].lower() in valid_exts
]
ref_image_path = random.choice(ref_image_paths)
image_bgr = cv2.imread(ref_image_path)
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(image_bgr)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).to("cuda")

print("✅ Generate images using reference face")
# ✅ Generate images using reference face
ip_adapter = IPAdapterFaceID(
    pipe,
    IP_ADAPTER_CKPT,
    "cuda"
)


prompt = "a photo of sksbbx beachybeach who is fair skinned smiling on the beach with thin black eyebrows"
negative_prompt = "blurry, distorted, low resolution, sunglasses"
for i in range(3):
    image = ip_adapter.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_embeds=faceid_embeds,
        num_samples=1,
        width=512,
        height=768,
        num_inference_steps=30,
        seed=i
    )
    image = image[0] if isinstance(image, list) else image
    image.show(title=f"sbsbbx beachybeach {i}")

    # Save the image
    os.makedirs("./output", exist_ok=True)
    image.save(f"{OUT_DIR}/sbsbbx_beachybeach_{i}.png")

print("✅ Generation complete.")

