import os
import io
import tarfile
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_pil_image
import glob
import webdataset as wds

input_dir = '/mnt/33t/cy/blip3o_dataset'
output_dir = '/mnt/33t/cy/mask_dataset'
data_files = glob.glob(os.path.join(input_dir, "*.tar"))

os.makedirs(output_dir, exist_ok=True)

# 加载数据集
for file in data_files:
    dataset = load_dataset("webdataset", data_files=file)

    # 输出 tar writer
    writers = {}
    def get_writer(shard_id):
        if shard_id not in writers:
            writers[shard_id] = wds.TarWriter(os.path.join(output_dir, os.path.basename(file)))
        return writers[shard_id]

    # 假设每个样本带 shard id
    for i, sample in enumerate(dataset['train']):
        img = sample["image"]  # PIL Image
        img_bytes_io = io.BytesIO()
        img.save(img_bytes_io, format="PNG")
        img_bytes = img_bytes_io.getvalue()

        # 得到 filename（假设 sample 中有 id 或 key）
        filename = f"{i:06d}"

        # 调用语义标注函数，保存 semantic json 并返回 semantic mask (H, W) tensor
        semantic_json_path = f"/tmp/{filename}_semantic.json"
        semantic_annotation_pipeline(
            filename=filename,
            data_path="/tmp/images/",  # 你需要实现写入 img 的逻辑
            output_path="/tmp/",
            rank=0,
            save_img=False,
            clip_processor=...,
            clip_model=...,
            oneformer_ade20k_processor=...,
            oneformer_ade20k_model=...,
            oneformer_coco_processor=...,
            oneformer_coco_model=...,
            blip_processor=...,
            blip_model=...,
            clipseg_processor=...,
            clipseg_model=...,
            mask_generator=...
        )

        # 加载 semantic_json，提取 mask（自行定义读取与转换函数）
        mask = extract_mask_from_json(semantic_json_path)  # H x W tensor

        # 压缩 mask 到 256 像素（保持比例）
        H, W = mask.shape
        ratio = (256.0 / (H * W)) ** 0.5
        new_size = (max(1, int(H * ratio)), max(1, int(W * ratio)))
        mask_resized = Resize(new_size, interpolation=Image.NEAREST)(to_pil_image(mask.unsqueeze(0).float())).convert("L")

        mask_bytes_io = io.BytesIO()
        mask_resized.save(mask_bytes_io, format="PNG")
        mask_bytes = mask_bytes_io.getvalue()

        # 写入 tar 包
        shard_id = i // 1000
        sample_key = filename
        writer = get_writer(shard_id)
        writer.write({
            f"{sample_key}.png": img_bytes,
            f"{sample_key}.mask.png": mask_bytes,
            f"{sample_key}.json": open(semantic_json_path, "rb").read()
        })

    # 关闭所有 writer
    for w in writers.values():
        w.close()
