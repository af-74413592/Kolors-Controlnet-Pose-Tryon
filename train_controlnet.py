from pathlib import Path
import sys
from PIL import Image
# from lion_pytorch import Lion
import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import transformers
import torch
import diffusers
from diffusers.models import AutoencoderKL
from kolors.models.unet_2d_condition import UNet2DConditionModel
from kolors.models.controlnet import ControlNetModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from huggingface_hub import create_repo, upload_folder
import math
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from torch.utils.data import Dataset
import os
import torch
import argparse
import shutil
from diffusers import DDPMScheduler
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from diffusers.utils import load_image
from kolors.pipelines.pipeline_controlnet_xl_kolors_img2img_face import StableDiffusionXLControlNetImg2ImgPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from diffusers.optimization import get_scheduler
from kolors.models.ipa_faceid_plus.ipa_faceid_plus import ProjPlusModel
from kolors.models.ipa_faceid_plus.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor

import wandb
parser = argparse.ArgumentParser(description='run kolors tryon controlnet')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--data_path', type=str, default="/workspace/train9.5",required=False)
parser.add_argument("--width",type=int,default=1536,)
parser.add_argument("--height",type=int,default=1024,)
parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
parser.add_argument('--guidance_scale', type=float, default=5.0, required=False)
parser.add_argument(
    "--scale_lr",
    action="store_true",
    default=False,
    help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument("--weight_decay",type=float,default=1e-2)
parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
parser.add_argument("--num_train_epochs", type=int, default=100)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=50000,
    help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
)
############################################################
parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="kolorstryoncontrolout",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
parser.add_argument(
    "--report_to",
    type=str,
    default="wandb",
    help=(
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
        ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    ),
)
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument(
    "--hub_model_id",
    type=str,
    default=None,
    help="The name of the repository to keep in sync with the local `output_dir`.",
)
parser.add_argument(
    "--logging_steps",
    type=int,
    default=500,
    help=(
        "logging_steps"
    ),
)
parser.add_argument(
    "--checkpointing_steps",
    type=int,
    default=500,
    help=(
        "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
        " training using `--resume_from_checkpoint`."
    ),
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=5,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default='bf16',
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)
parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
parser.add_argument(
    "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
)

parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="kolorstryoncontrol2025",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
)
parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default="1.0",
    )
parser.add_argument(
        "--control_guidance_end",
        type=float,
        default="0.9",
    )
parser.add_argument(
        "--ip_scale",
        type=float,
        default="0.5",
    )


args = parser.parse_args()

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

data_path = args.data_path

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

def collate_fn(batch):
    cond_imgs = [item['cond_img'] for item in batch]
    label_imgs = [item['label_img'] for item in batch]
    person_imgs = [item['person_img'] for item in batch]
    person_prompts = [item['person_prompt'] for item in batch]
    return cond_imgs, label_imgs,person_imgs, person_prompts

def log_validation(pipe, args, accelerator,cond_img,label_img,person_prompt,crop_image,face_embeds):
    logger.info("Running validation... ")
    import random
    import time
    random.seed(time.time())
    seed = random.randint(0, 2147483647)
    generator = torch.manual_seed(seed)

    with torch.no_grad():

        sample_image = pipe(
            prompt= person_prompt ,
            # image = init_image,
            controlnet_conditioning_scale = args.controlnet_conditioning_scale,
            control_guidance_end = args.control_guidance_end, 
            # ip_adapter_image=[ ip_adapter_img ],
            face_crop_image = crop_image,
            face_insightface_embeds = face_embeds,
            strength= 1.0 , 
            control_image = cond_img,
            negative_prompt= 'nsfw，脸部阴影，低分辨率，糟糕的解剖结构、糟糕的手，缺失手指、质量最差、低质量、jpeg伪影、模糊、糟糕，黑脸，霓虹灯' , 
            num_inference_steps= 20, 
            guidance_scale= args.guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
        ).images[0]

        image_logs = []
        image_logs.append({
                        "control_img": cond_img, 
                        "label_img" : label_img,
                        "samples": sample_image,
                        "crop_image": crop_image, 
                        "prompt": person_prompt,
                        })

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                formatted_images = []
                for log in image_logs:
                    formatted_images.append(wandb.Image(log["control_img"], caption="control images"))
                    formatted_images.append(wandb.Image(log["label_img"], caption="label images"))
                    formatted_images.append(wandb.Image(log["crop_image"], caption="face images"))
                    formatted_images.append(wandb.Image(log["samples"], caption=log["prompt"]))
                tracker.log({'test_image': formatted_images})
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

class KolorsDataset(Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.person_path = os.path.join(self.data_path, "person")
        self.cloth_path = os.path.join(self.data_path, "cloth")
        self.pose_path = os.path.join(self.data_path, "pose")
        self.person_txt_path = os.path.join(self.data_path, "persontxts")
        # self.cloth_txt_path = os.path.join(self.data_path, "clothtxts")

        self.person_paths = os.listdir(self.person_path)
        self.cloth_paths = os.listdir(self.cloth_path)
        self.pose_paths = os.listdir(self.pose_path)
        self.person_txt_paths = os.listdir(self.person_txt_path)
        # self.cloth_txt_paths = os.listdir(self.cloth_txt_path)


    def __len__(self):
        return len(self.person_paths)

    def __getitem__(self, idx):
        assert len(self.person_paths) == len(self.cloth_paths) == len(self.pose_paths) == len(self.person_txt_paths) 
        person_path = os.path.join(self.person_path, self.person_paths[idx])
        cloth_path = os.path.join(self.cloth_path, self.person_paths[idx])
        pose_path = os.path.join(self.pose_path, self.person_paths[idx])
        cloth_img = Image.open(cloth_path)
        assert cloth_img.size == (768, 1024)
        person_img = Image.open(person_path)
        assert person_img.size == (768, 1024)
        pose_img = Image.open(pose_path)
        assert pose_img.size == (768, 1024)
        #横向拼接pose和cloth
        cond_img = Image.fromarray(np.concatenate((np.array(pose_img), np.array(cloth_img)), axis=1))
        #横向拼接person和cloth
        label_img = Image.fromarray(np.concatenate((np.array(person_img), np.array(cloth_img)), axis=1))
        with open(os.path.join(self.person_txt_path, self.person_paths[idx][:-4]+'.txt'), 'r',encoding='utf-8') as f1:
            person_prompt = f1.read().split('\n')[0].strip()
        # with open(os.path.join(self.cloth_txt_path, self.person_paths[idx][:-4]+'.txt'), 'r',encoding='utf-8') as f2:
        #     cloth_prompt = f2.read().split('\n')[0].strip()

        return {'label_img':label_img,
                'cond_img':cond_img,
                'person_img' :person_img,
                'person_prompt':person_prompt,
                } 

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



if __name__ == '__main__':
    ckpt_dir = 'weights/Kolors'
    controlnet_dir = 'weights/Kolors-ControlNet-Pose'
    ip_model_dir = 'weights/Kolors-IP-Adapter-FaceID-Plus'
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)
        
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # if args.use_ema:
    #     ema_unet = UNet2DConditionModel.from_pretrained(
    #         args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    #     )
    #     ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warn(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")
            
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    # # Initialize the optimizer
    # if args.use_8bit_adam:
    #     try:
    #         import bitsandbytes as bnb
    #     except ImportError:
    #         raise ImportError(
    #             "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
    #         )

    #     optimizer_cls = bnb.optim.AdamW8bit
    # optimizer_cls = Lion
    # optimizer_cls = torch.optim.SGD
    # optimizer_cls = torch.optim.AdamW
    import bitsandbytes as bnb
    optimizer_cls = bnb.optim.AdamW8bit
    

    text_encoder = ChatGLMModel.from_pretrained( f'{ckpt_dir}/text_encoder')
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    noise_scheduler = DDPMScheduler.from_pretrained(ckpt_dir, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(f'{ckpt_dir}/vae', subfolder = "vae", revision = None)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    unet = UNet2DConditionModel.from_pretrained(f'{ckpt_dir}/unet', subfolder = "unet", revision = None)
    controlnet = ControlNetModel.from_pretrained( controlnet_dir , revision=None)

    #### clip image encoder for face structure
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(f'{ip_model_dir}/clip-vit-large-patch14-336', ignore_mismatched_sizes=True)
    clip_image_encoder.to(accelerator.device)
    clip_image_processor = CLIPImageProcessor(size = 336, crop_size = 336)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device) 
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    
    text_encoder.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)

    face_info_generator = FaceInfoGenerator(root_dir = "./")

    train_dataset = KolorsDataset(data_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print(args.num_train_epochs)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num max_train_steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    
    pipe = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=vae,
            controlnet = controlnet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            force_zeros_for_empty_prompt=False,
            face_clip_encoder=clip_image_encoder,
            face_clip_processor=clip_image_processor,
            )

    pipe = pipe.to(accelerator.device)
    pipe.enable_model_cpu_offload()

    pipe.load_ip_adapter_faceid_plus(f'{ip_model_dir}/ipa-faceid-plus.bin', device = accelerator.device)

    scale = args.ip_scale
    pipe.set_face_fidelity_scale(scale)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            cond_imgs,lable_imgs,person_imgs,person_prompts = batch
            crop_images = []
            face_embeds = []
            for person_img in person_imgs: 
                face_info = face_info_generator.get_faceinfo_one_img(person_img)
                if face_info is not None:
                    face_bbox_square = face_bbox_to_square(face_info["bbox"])
                    crop_image = person_img.crop(face_bbox_square)
                    crop_image = crop_image.resize((336, 336))
                    crop_images.append(crop_image)

                    face_embed = torch.from_numpy(np.array([face_info["embedding"]]))
                    face_embed = face_embed.to(accelerator.device, dtype = weight_dtype)
                    face_embeds.append(face_embed)
                else:
                    crop_images.append(Image.new('RGB', (336, 336), color = 'white'))
                    face_embeds.append(torch.zeros((1, 512)).to(accelerator.device, dtype = weight_dtype))
            if global_step % args.logging_steps == 0:
                log_validation(pipe,args,accelerator,cond_imgs[0],lable_imgs[0],person_prompts[0],crop_images[0],face_embeds[0])

            # Gather the losses across all processes for logging (if we use distributed training).

            
            loss = pipe.train_step(accelerator,optimizer,lr_scheduler,
                                   height=1024, width=1536,
                                    prompt=person_prompts,
                                    controlnet_conditioning_scale = args.controlnet_conditioning_scale,
                                    control_guidance_end = args.control_guidance_end, 
                                    num_images_per_prompt = 1,
                                    face_crop_image= crop_images,
                                    ori_image=lable_imgs, 
                                    control_image = cond_imgs,
                                    face_insightface_embeds= face_embeds,
                                    )

            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            epoch_loss += train_loss
            
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    # if args.checkpoints_total_limit is not None:
                    #     checkpoints = os.listdir(args.output_dir)
                    #     checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    #     checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    #     # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    #     if len(checkpoints) >= args.checkpoints_total_limit:
                    #         num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    #         removing_checkpoints = checkpoints[0:num_to_remove]

                    #         logger.info(
                    #             f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    #         )
                    #         logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    #         for removing_checkpoint in removing_checkpoints:
                    #             removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    #             shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    controlnet.save_pretrained(os.path.join(save_path, "controlnet"))
                    logger.info(f"Saved state to {save_path}")

            # logs = {"step_loss": loss.detach().item(), "lr": model.lr_scheduler.get_last_lr()[0]}
            logs = {"step_loss": loss.detach().item(), "lr": args.learning_rate}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        epoch_loss /= step
        accelerator.log({"epoch_loss": epoch_loss}, step=epoch)


