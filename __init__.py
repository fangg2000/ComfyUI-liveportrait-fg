import os
import sys
import numpy as np
import time
import re
import torch
import threading
import random
import io

import os.path as osp
import tyro
import subprocess
import cv2

from .liveportrait.src.config.argument_config import ArgumentConfig
from .liveportrait.src.config.inference_config import InferenceConfig
from .liveportrait.src.config.crop_config import CropConfig
from .liveportrait.src.live_portrait_pipeline import LivePortraitPipeline
from .liveportrait.src.live_portrait_pipeline_animal import LivePortraitPipelineAnimal
from torchvision import transforms
from datetime import datetime
from PIL import Image
from pathlib import Path


# 动态获取项目根目录（入口文件在 ComfyUI-liveportrait-fg 下）
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

now_dir = os.path.dirname(os.path.abspath(__file__))

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def video_to_tensor(video_path, target_size=(224, 224), num_frames=16, sample_rate=1):
    """
    将视频文件转换为 PyTorch 张量
    参数:
        video_path (str): 视频文件路径
        target_size (tuple): 目标分辨率 (H, W)
        num_frames (int): 提取的帧数
        sample_rate (int): 采样间隔（每隔 sample_rate 帧取一帧）
    返回:
        tensor (torch.Tensor): 形状为 (T, C, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法读取视频文件: {video_path}")

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            # 调整分辨率
            frame = cv2.resize(frame, target_size)
            # BGR 转 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        frame_count += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError("视频中没有有效帧")

    # 转换为 NumPy 数组 (T, H, W, C)
    frames_array = np.stack(frames, axis=0)

    return frames_array

def numpy_to_tensor(frames_array):
    """
    将 NumPy 数组转换为 PyTorch 张量
    参数:
        frames_array (np.ndarray): 形状为 (T, H, W, C)
    返回:
        tensor (torch.Tensor): 形状为 (T, C, H, W)
    """
    # 转换为张量并调整维度
    tensor = torch.from_numpy(frames_array).float() / 255.0  # 归一化到 [0, 1]
    tensor = tensor.permute(0, 3, 1, 2)  # (T, C, H, W)

    return tensor

def get_comfyui_root():
    # 获取主模块（通常是启动脚本，如main.py）
    main_module = sys.modules.get('__main__')
    if main_module and hasattr(main_module, '__file__'):
        main_path = os.path.abspath(main_module.__file__)
        root_dir = os.path.dirname(main_path)
        return root_dir
    else:
        # 若无法获取主模块路径，回退到当前工作目录
        return now_dir

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")

class LiveportraitType:
    CATEGORY = "fg/LivePortrait type Node"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    def __init__(self):
        self.save_dir = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "driving_type": (['human', 'animal'], {
                    "default": "human"
                }),
            },
        }

    def generate(self, driving_type):
        return (driving_type,)

# 注册节点
class LiveportraitInit:
    CATEGORY = "fg/LivePortrait init Node"
    RETURN_TYPES = ("DRIVING",)
    RETURN_NAMES = ("driving",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    def __init__(self):
        self.save_dir = None

    @classmethod
    def INPUT_TYPES(cls):
        driving_list = [""]

        # 遍历目录及其子目录
        for root, dirs, files in os.walk(now_dir+'/liveportrait/assets/examples/driving'):
            for file in files:
                if file.endswith("_fg.pkl"):
                    # print(f'匹配文件--{file}')
                    driving_list.append(str(file).replace('_fg.pkl', ''))

        return {
            "required": {
            },
            "optional": {
                "images": ("IMAGE",),  # 输入视频张量 [T, H, W, C]
                "driving": (driving_list, {
                    "default": ""
                }),
                "save_driving_name": ("STRING", {"default": ""}),
                "flag_use_half_precision": ("BOOLEAN", {"default": True}),
                "flag_crop_driving_video": ("BOOLEAN", {"default": False}),
                "device_id": ("INT", {"default": 0}),
                "flag_force_cpu": ("BOOLEAN", {"default": False}),
                "flag_normalize_lip": ("BOOLEAN", {"default": False}),
                "flag_source_video_eye_retargeting": ("BOOLEAN", {"default": False}),
                "flag_eye_retargeting": ("BOOLEAN", {"default": False}),
                "flag_lip_retargeting": ("BOOLEAN", {"default": False}),
                "flag_stitching": ("BOOLEAN", {"default": True}),
                "flag_relative_motion": ("BOOLEAN", {"default": True}),
                "flag_pasteback": ("BOOLEAN", {"default": True}),
                "flag_do_crop": ("BOOLEAN", {"default": True}),
                "driving_option": (["expression-friendly", "pose-friendly"], {"default": "expression-friendly"}),
                "driving_multiplier": ("FLOAT", {"default": 1.0}),
                "driving_smooth_observation_variance": ("FLOAT", {"default": 3e-7}),
                "audio_priority": (["source", "driving"], {"default": "driving"}),
                "animation_region": (["exp", "pose", "lip", "eyes", "all"], {"default": "all"}),
                "det_thresh": ("FLOAT", {"default": 0.15}),
                "scale": ("FLOAT", {"default": 2.3}),
                "vx_ratio": ("FLOAT", {"default": 0}),
                "vy_ratio": ("FLOAT", {"default": -0.125}),
                "flag_do_rot": ("BOOLEAN", {"default": True}),
                "source_max_dim": ("INT", {"default": 1280}),
                "source_division": ("INT", {"default": 2}),
            },
        }

    def generate(self,
                 images,
                 driving,
                 save_driving_name,
                 flag_use_half_precision=True,
                 flag_crop_driving_video=False,
                 device_id=0,
                 flag_force_cpu=False,
                 flag_normalize_lip=False,
                 flag_source_video_eye_retargeting=False,
                 flag_eye_retargeting=False,
                 flag_lip_retargeting=False,
                 flag_stitching=True,
                 flag_relative_motion=True,
                 flag_pasteback=True,
                 flag_do_crop=True,
                 driving_option="expression-friendly",
                 driving_multiplier=1.0,
                 driving_smooth_observation_variance=3e-7,
                 audio_priority="driving",
                 animation_region="all",
                 det_thresh=0.15,
                 scale=2.3,
                 vx_ratio=0,
                 vy_ratio=-0.125,
                 flag_do_rot=True,
                 source_max_dim=1280,
                 source_division=2):
        # 取根目录
        root_path = get_comfyui_root()
        print(f'根目录a--{root_path}')
        print(f'模板目录b--{now_dir}')

        # set tyro theme
        tyro.extras.set_accent_color("bright_cyan")
        args = tyro.cli(ArgumentConfig)

        # print(f'驱动视频a--{args.driving}')
        # args.source = root_path + image_path

        args.driving_name = save_driving_name

        # 添加所有参数
        args.flag_use_half_precision = flag_use_half_precision
        args.flag_crop_driving_video = flag_crop_driving_video
        args.device_id = device_id
        args.flag_force_cpu = flag_force_cpu
        args.flag_normalize_lip = flag_normalize_lip
        args.flag_source_video_eye_retargeting = flag_source_video_eye_retargeting
        args.flag_eye_retargeting = flag_eye_retargeting
        args.flag_lip_retargeting = flag_lip_retargeting
        args.flag_stitching = flag_stitching
        args.flag_relative_motion = flag_relative_motion
        args.flag_pasteback = flag_pasteback
        args.flag_do_crop = flag_do_crop
        args.driving_option = driving_option
        args.driving_multiplier = driving_multiplier
        args.driving_smooth_observation_variance = driving_smooth_observation_variance
        args.audio_priority = audio_priority
        args.animation_region = animation_region
        args.det_thresh = det_thresh
        args.scale = scale
        args.vx_ratio = vx_ratio
        args.vy_ratio = vy_ratio
        args.flag_do_rot = flag_do_rot
        args.source_max_dim = source_max_dim
        args.source_division = source_division

        if driving != '':
            driving_path = f'{now_dir}{args.driving_path}/{driving}_fg.pkl'
            # print(f'输出路径--{driving_path}')
            return (driving_path,)
        else:
            # 初始化视频写入器
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = root_path + f"/temp/lp_tmp_{timestamp}.mp4"

            # 确保输入张量是4维的 [batch, H, W, C]
            if images.dim() != 4:
                raise ValueError(f"输入张量维度错误，应为 [B, H, W, C]，实际为 {images.shape}")

            # 创建输出目录
            # os.makedirs(output_dir, exist_ok=True)
            # output_path = os.path.join(output_dir, filename)

            # 获取视频参数
            num_frames, height, width, _ = images.shape
            codec = "mp4v"  # MP4编码器（H.264需额外配置）

            # 初始化VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(video_path, fourcc, 24, (width, height))

            if not video_writer.isOpened():
                raise RuntimeError(f"无法创建视频文件: {video_path}")

            # 逐帧处理并写入
            for frame in images:
                # 转换张量到CPU和numpy，并调整数值范围到 [0, 255]
                frame_np = frame.cpu().numpy() * 255.0
                frame_np = frame_np.astype(np.uint8)

                # 转换颜色空间 RGB -> BGR（OpenCV默认）
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                # 写入帧
                video_writer.write(frame_bgr)

            video_writer.release()
            args.driving = video_path

        ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
        if osp.exists(ffmpeg_dir):
            os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

        if not fast_check_ffmpeg():
            raise ImportError(
                "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
            )

        fast_check_args(args)

        # specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        args.root_path = root_path
        # 用于保存.pkl模板文件
        args.dir_path = now_dir
        inference_cfg.root_path = root_path
        crop_cfg.root_path = root_path
        result_driving = None

        # print(f'人物初始化参数b--{args.driving}')
        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
        # run
        result_driving = live_portrait_pipeline.execute(args)
        print(f'模板保存路径--{result_driving}')

        # 删除缓存文件
        os.remove(video_path)
        # 刷新列表
        self.INPUT_TYPES()

        return (result_driving,)

class LiveportraitDriving:
    CATEGORY = "fg/LivePortrait driving Node"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "video_path",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    def __init__(self):
        self.save_dir = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "driving": ("DRIVING",),
                "image": ("IMAGE",),
            },
            "optional": {
                "driving_type": (['human', 'animal'], {
                    "default": "human"
                }),
                "scale_crop_driving_video": ("FLOAT", {"default": 2.2}),
                "vx_ratio_crop_driving_video": ("FLOAT", {"default": 0.}),
                "vy_ratio_crop_driving_video": ("FLOAT", {"default": -0.1}),
            },
        }

    def generate(self,
                 driving,
                 image,
                 driving_type,
                 scale_crop_driving_video,
                 vx_ratio_crop_driving_video,
                 vy_ratio_crop_driving_video,
        ):
        # print(f'参数driving路径--{driving}')
        # 取根目录
        root_path = get_comfyui_root()

        # set tyro theme
        tyro.extras.set_accent_color("bright_cyan")
        args = tyro.cli(ArgumentConfig)
        args.scale_crop_driving_video = scale_crop_driving_video
        args.vx_ratio_crop_driving_video = vx_ratio_crop_driving_video
        args.vy_ratio_crop_driving_video = vy_ratio_crop_driving_video

        # 处理输入的图片数据
        # 将张量转换为 PIL 图像
        pil_image = tensor2pil(image)

        # 保存为图像文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = root_path + f"/temp/lp_tmp_{timestamp}.png"
        pil_image.save(img_path)

        # print(f'驱动视频a--{args.driving}')
        args.source = img_path
        args.driving = driving

        ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
        if osp.exists(ffmpeg_dir):
            os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

        if not fast_check_ffmpeg():
            raise ImportError(
                "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
            )

        fast_check_args(args)

        # specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        args.root_path = root_path
        inference_cfg.root_path = root_path
        crop_cfg.root_path = root_path
        result_video = None

        if 'animal' == driving_type:
            print(f'动物初始化参数b--{args.driving}')
            sys.path.append(root_path)

            live_portrait_pipeline_animal = LivePortraitPipelineAnimal(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg
            )
            # run
            result_video, v_contact, v_git = live_portrait_pipeline_animal.execute(args)
        else:
            print(f'人物初始化参数b--{args.driving}')
            live_portrait_pipeline = LivePortraitPipeline(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg
            )
            # run
            result_video, v_contact = live_portrait_pipeline.execute(args)

        print(f'视频返回结果--{result_video}')

        # 删除缓存文件
        os.remove(img_path)

        result_path = root_path+'/'+result_video

        # 处理视频数据
        # 加载视频帧
        cap = cv2.VideoCapture(result_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为 RGB 并归一化
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frames.append(frame_tensor)
        cap.release()

        if not frames:
            raise ValueError("视频未包含有效帧")

        # 合并帧张量 [T, H, W, C]
        video_tensor = torch.stack(frames, dim=0)

        # return (numpy_to_tensor(video_to_tensor(video_path=root_path+'/'+result_video)),)
        return (video_tensor, result_path,)

NODE_CLASS_MAPPINGS = {
    "LiveportraitType": LiveportraitType,
    "LiveportraitInit": LiveportraitInit,
    "LiveportraitDriving": LiveportraitDriving,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LiveportraitInit": "liveportrait init",
    "LiveportraitDriving": "liveportrait driving video",
    # "TranslateNode": "translate/翻译(中英俄)",
}

# 主程序入口 ---------------------------------------------------------
# if __name__ == "__main__":
#     print("系统已启动".center(40))
    # testNode = LiveportraitInit()
    # testNode.generate('test', '', '/liveportrait/assets/examples/driving/d9.mp4', True)

    # testNode = LiveportraitDriving()
    # testNode.generate('human',
    #                   '/liveportrait/assets/examples/source/s6.jpg',
    #                   '/liveportrait/assets/examples/driving/test_fg.pkl')
