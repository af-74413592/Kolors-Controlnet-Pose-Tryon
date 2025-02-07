import torch
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from diffusers.utils import load_image
import os,sys
import gradio as gr

from kolors.pipelines.pipeline_controlnet_xl_kolors_img2img_face import StableDiffusionXLControlNetImg2ImgPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.models.controlnet import ControlNetModel

from diffusers import  AutoencoderKL
from kolors.models.unet_2d_condition import UNet2DConditionModel

from diffusers import EulerDiscreteScheduler
from PIL import Image
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

example_path = os.path.join(os.path.dirname(__file__), 'examples')


class FaceInfoGenerator():
    def __init__(self, root_dir = "./"):
        self.app = FaceAnalysis(name = 'antelopev2', root = root_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id = 0, det_size = (640, 640))

    def get_faceinfo_one_img(self, face_image):
        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))

        if len(face_info) == 0:
            face_info = None
        else:
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
        return face_info
    
def face_bbox_to_square(bbox):
    ## l, t, r, b to square l, t, r, b
    l,t,r,b = bbox
    cent_x = (l + r) / 2
    cent_y = (t + b) / 2
    w, h = r - l, b - t
    r = max(w, h) / 2

    l0 = cent_x - r
    r0 = cent_x + r
    t0 = cent_y - r
    b0 = cent_y + r

    return [l0, t0, r0, b0]


ckpt_dir = f'weights/Kolors'
text_encoder = ChatGLMModel.from_pretrained(
    f'{ckpt_dir}/text_encoder').to(dtype=torch.bfloat16)
tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).to(dtype=torch.bfloat16)
scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).to(dtype=torch.bfloat16)

control_path = f'weights/Kolors-Controlnet-Pose-Tryon'
controlnet = ControlNetModel.from_pretrained( control_path , revision=None).to(dtype=torch.bfloat16)

face_info_generator = FaceInfoGenerator(root_dir = "./")

clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(f'weights/Kolors-IP-Adapter-FaceID-Plus/clip-vit-large-patch14-336', ignore_mismatched_sizes=True)
clip_image_encoder.to('cuda')
clip_image_processor = CLIPImageProcessor(size = 336, crop_size = 336)

pipe = StableDiffusionXLControlNetImg2ImgPipeline(
        vae=vae,
        controlnet = controlnet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        # image_encoder=image_encoder,
        # feature_extractor=clip_image_processor,
        force_zeros_for_empty_prompt=False,
        face_clip_encoder=clip_image_encoder,
        face_clip_processor=clip_image_processor,
        )
if hasattr(pipe.unet, 'encoder_hid_proj'):
    pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj
ip_scale = 0.5
pipe.load_ip_adapter_faceid_plus(f'weights/Kolors-IP-Adapter-FaceID-Plus/ipa-faceid-plus.bin', device = 'cuda')
pipe.set_face_fidelity_scale(ip_scale)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

def infer(face_img,pose_img, garm_img, prompt,negative_prompt, n_samples, n_steps, seed):
    face_img = Image.open(face_img)
    pose_img = Image.open(pose_img)
    garm_img = Image.open(garm_img)
    face_img = face_img.resize((336, 336))
    pose_img = pose_img.resize((768, 1024))
    garm_img = garm_img.resize((768, 1024))
    
    background = Image.new("RGB", (768, 768), (255, 255, 255))
    #将face_img粘贴到background中心
    background.paste(face_img, (int((768 - 336) / 2), int((768 - 336) / 2)))

    face_info = face_info_generator.get_faceinfo_one_img(background)

    face_embeds = torch.from_numpy(np.array([face_info["embedding"]]))
    face_embeds = face_embeds.to('cuda', dtype = torch.bfloat16)    
        
    controlnet_conditioning_scale = 1.0
    control_guidance_end = 0.9
    #strength 越是小，则生成图片越是依赖原始图片。
    strength = 1.0
        
    im1 = np.array(pose_img)
    im2 = np.array(garm_img)

    condi_img = Image.fromarray( np.concatenate( (im1, im2), axis=1 ) )
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt= prompt ,
        # image = init_image,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        control_guidance_end = control_guidance_end, 
        # ip_adapter_image=[ ip_adapter_img ],
        face_crop_image = face_img,
        face_insightface_embeds = face_embeds,
        strength= strength , 
        control_image = condi_img,
        negative_prompt= negative_prompt , 
        num_inference_steps=n_steps , 
        guidance_scale= 5.0,
        num_images_per_prompt=n_samples,
        generator=generator,
    ).images
    return image


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("# KolorsControlnerTryon Demo")
    with gr.Row():
        with gr.Column():
            pose_img = gr.Image(label="Pose", sources='upload', type="filepath", height=768, value=os.path.join(example_path, 'pose/1.jpg'))
            example = gr.Examples(
                inputs=pose_img,
                examples_per_page=10,
                examples=[
                    os.path.join(example_path, 'pose/1.jpg'),
                    os.path.join(example_path, 'pose/2.jpg'),
                    os.path.join(example_path, 'pose/3.jpg'),
                    os.path.join(example_path, 'pose/4.jpg'),
                    os.path.join(example_path, 'pose/5.jpg'),
                    os.path.join(example_path, 'pose/6.jpg'),
                    os.path.join(example_path, 'pose/7.jpg'),
                    os.path.join(example_path, 'pose/8.jpg'),
                    os.path.join(example_path, 'pose/9.jpg'),
                    os.path.join(example_path, 'pose/10.jpg'),
                ])
        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="filepath", height=768, value=os.path.join(example_path, 'garment/1.jpg'),)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=10,
                examples=[
                    os.path.join(example_path, 'garment/1.jpg'),
                    os.path.join(example_path, 'garment/2.jpg'),
                    os.path.join(example_path, 'garment/3.jpg'),
                    os.path.join(example_path, 'garment/4.jpg'),
                    os.path.join(example_path, 'garment/5.jpg'),
                    os.path.join(example_path, 'garment/6.jpg'),
                    os.path.join(example_path, 'garment/7.jpg'),
                    os.path.join(example_path, 'garment/8.jpg'),
                    os.path.join(example_path, 'garment/9.jpg'),
                    os.path.join(example_path, 'garment/10.jpg'),
                ])
    with gr.Row():
        with gr.Column():
            face_img = gr.Image(label="Face", sources='upload', type="filepath", height=336, value=os.path.join(example_path, 'face/1.png'),)
            example = gr.Examples(
                inputs=face_img,
                examples_per_page=10,
                examples=[
                    os.path.join(example_path, 'face/1.png'),
                    os.path.join(example_path, 'face/2.png'),
                    os.path.join(example_path, 'face/3.png'),
                    os.path.join(example_path, 'face/4.png'),
                    os.path.join(example_path, 'face/5.png'),
                    os.path.join(example_path, 'face/6.png'),
                    os.path.join(example_path, 'face/7.png'),
                    os.path.join(example_path, 'face/8.png'),
                    os.path.join(example_path, 'face/9.png'),
                    os.path.join(example_path, 'face/10.png'),
                ])
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        prompt = gr.Textbox(value="这张图片上的模特穿着一件黑色的长袖T恤，T恤上印着彩色的字母'OBEY'。她还穿着一条牛仔裤。", show_label=False, elem_id="prompt")
        negative_prompt = gr.Textbox(value="nsfw，脸部阴影，低分辨率，糟糕的解剖结构、糟糕的手，缺失手指、质量最差、低质量、jpeg伪影、模糊、糟糕，黑脸，霓虹灯", show_label=False, elem_id="negative_prompt")
        n_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
        n_steps = gr.Slider(label="Steps", minimum=20, maximum=50, value=20, step=1)
        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
        run_button = gr.Button(value="Run")
    ips = [face_img,pose_img, garm_img, prompt,negative_prompt, n_samples, n_steps, seed]
    run_button.click(fn=infer, inputs=ips, outputs=[result_gallery])

block.launch(server_name='0.0.0.0', server_port=7865)
