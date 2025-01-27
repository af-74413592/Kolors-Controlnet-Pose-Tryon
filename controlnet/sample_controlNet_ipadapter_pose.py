import torch
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from diffusers.utils import load_image
import os,sys

# from kolors.pipelines.pipeline_controlnet_xl_kolors_img2img import StableDiffusionXLControlNetImg2ImgPipeline
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

from annotator.midas import MidasDetector
from annotator.util import resize_image,HWC3
from annotator.dwpose import DWposeDetector
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

class FaceInfoGenerator():
    def __init__(self, root_dir = "./"):
        self.app = FaceAnalysis(name = 'antelopev2', root = root_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id = 0, det_size = (640, 640))

    def get_faceinfo_one_img(self, image_path):
        face_image = load_image(image_path)
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


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def process_canny_condition( image, canny_threods=[100,200] ):
    np_image = image.copy()
    np_image = cv2.Canny(np_image, canny_threods[0], canny_threods[1])
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    np_image = HWC3(np_image)
    return Image.fromarray(np_image)


model_midas = None
def process_depth_condition_midas(img, res = 1024):
    h,w,_ = img.shape
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        model_midas = MidasDetector()
    
    result = HWC3( model_midas(img) )
    result = cv2.resize( result, (w,h) )
    return Image.fromarray(result)

model_dwpose = None
def process_dwpose_condition( image, res=1024 ):
    h,w,_ = image.shape
    img = resize_image(HWC3(image), res)
    global model_dwpose
    if model_dwpose is None:
        model_dwpose = DWposeDetector()
    out_res, out_img = model_dwpose(image) 
    result = HWC3( out_img )
    result = cv2.resize( result, (w,h) )
    return Image.fromarray(result)


def infer( ip_image_path='./00001_00.jpg',  prompt='图片上的模特穿着白色LevisT恤和黑色的皮裤，搭配银色的耳环。', model_type = 'Pose' ):

    ckpt_dir = f'{root_dir}/weights/Kolors'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

    control_path = f'{root_dir}/weights/Kolors-ControlNet-Pose'
    controlnet = ControlNetModel.from_pretrained( control_path , revision=None).half()

    face_info_generator = FaceInfoGenerator(root_dir = "./")
    img = Image.open(ip_image_path)
    face_info = face_info_generator.get_faceinfo_one_img(ip_image_path)

    face_bbox_square = face_bbox_to_square(face_info["bbox"])
    crop_image = img.crop(face_bbox_square)
    crop_image = crop_image.resize((336, 336))
    crop_image = [crop_image]

    face_embeds = torch.from_numpy(np.array([face_info["embedding"]]))
    face_embeds = face_embeds.to('cuda', dtype = torch.float16)

    # IP-Adapter model
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained( f'{root_dir}/weights/Kolors-IP-Adapter-Plus/image_encoder',  ignore_mismatched_sizes=True).to(dtype=torch.float16)
    # ip_img_size = 336
    # clip_image_processor = CLIPImageProcessor( size=ip_img_size, crop_size=ip_img_size )

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
    # pipe.load_ip_adapter( f'{root_dir}/weights/Kolors-IP-Adapter-Plus' , subfolder="", weight_name=["ip_adapter_plus_general.bin"])
    pipe.load_ip_adapter_faceid_plus(f'weights/Kolors-IP-Adapter-FaceID-Plus/ipa-faceid-plus.bin', device = 'cuda')
    pipe.set_face_fidelity_scale(ip_scale)
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    negative_prompt = 'nsfw，脸部阴影，低分辨率，糟糕的解剖结构、糟糕的手，缺失手指、质量最差、低质量、jpeg伪影、模糊、糟糕，黑脸，霓虹灯'
    
    MAX_IMG_SIZE=1024
    controlnet_conditioning_scale = 1.0
    control_guidance_end = 0.9
    #strength 越是小，则生成图片越是依赖原始图片。
    strength = 1.0
    
    
    basename = ip_image_path.rsplit('/',1)[-1].rsplit('.',1)[0]

    init_image = Image.open( ip_image_path )
    
    init_image = resize_image( init_image,  MAX_IMG_SIZE)
    
    if model_type == 'Canny':
        condi_img = process_canny_condition( np.array(init_image) )
    elif model_type == 'Depth':
        condi_img = process_depth_condition_midas( np.array(init_image), MAX_IMG_SIZE )
    elif model_type == 'Pose':
        condi_img = process_dwpose_condition( np.array(init_image), MAX_IMG_SIZE)

    # ip_adapter_img = Image.open(ip_image_path)
    # pipe.set_ip_adapter_scale([ ip_scale ])
    
    generator = torch.Generator(device="cpu").manual_seed(66)
    image = pipe(
        prompt= prompt ,
        # image = init_image,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        control_guidance_end = control_guidance_end, 
        # ip_adapter_image=[ ip_adapter_img ],
        face_crop_image = crop_image,
        face_insightface_embeds = face_embeds,
        strength= strength , 
        control_image = condi_img,
        negative_prompt= negative_prompt , 
        num_inference_steps= 50 , 
        guidance_scale= 5.0,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    
    image.save(f'{root_dir}/controlnet/outputs/{model_type}_ipadapter_{basename}.jpg')
    condi_img.save(f'{root_dir}/controlnet/outputs/{model_type}_{basename}_condition.jpg')


if __name__ == '__main__':
    import fire
    fire.Fire(infer)
    









