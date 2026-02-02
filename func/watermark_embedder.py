# -*- coding: utf-8 -*-
"""
水印嵌入模块 - 提供图像水印嵌入功能
支持单图像、文件夹批量处理，以及大图像滑动窗口处理
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, List, Tuple
from pathlib import Path
from tqdm import tqdm

from .watermark_core import WatermarkCore


class WatermarkEmbedder:
    """水印嵌入器，支持多种输入形式和大图像处理"""
    
    # 支持的图像格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, 
                 watermark_secret: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化水印嵌入器
        
        Args:
            watermark_secret: 30位二进制水印密钥，如果为None则随机生成
            device: 运行设备
        """
        self.core = WatermarkCore(device=device)
        self.device = device
        self.window_size = WatermarkCore.SUPPORTED_SIZE  # 200
        
        # 生成水印密钥和图案
        self.secret_input = self.core.generate_secret(watermark_secret)
        self.secret_pattern = self.core.encode_secret(self.secret_input)
        self.secret_string = self.core.secret_to_string(self.secret_input)
        
    def get_secret_string(self) -> str:
        """获取当前水印密钥字符串"""
        return self.secret_string
    
    def set_secret(self, watermark_secret: str):
        """
        设置新的水印密钥
        
        Args:
            watermark_secret: 30位二进制水印密钥
        """
        self.secret_input = self.core.generate_secret(watermark_secret)
        self.secret_pattern = self.core.encode_secret(self.secret_input)
        self.secret_string = self.core.secret_to_string(self.secret_input)
    
    def _pad_image_center(self, image: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        将小于200x200的图像填充到200x200，原图居中
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[Image.Image, Tuple]: (填充后的图像, 填充信息(left, top, right, bottom))
        """
        w, h = image.size
        target_size = self.window_size
        
        # 计算填充量
        pad_w = target_size - w
        pad_h = target_size - h
        
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        
        # 创建填充后的图像（使用灰色填充）
        padded = Image.new('RGB', (target_size, target_size), (128, 128, 128))
        padded.paste(image, (left, top))
        
        return padded, (left, top, right, bottom)
    
    def _unpad_image(self, 
                     image: Image.Image, 
                     original_size: Tuple[int, int],
                     pad_info: Tuple[int, int, int, int]) -> Image.Image:
        """
        去除填充，恢复原始大小
        
        Args:
            image: 填充后的图像
            original_size: 原始大小 (width, height)
            pad_info: 填充信息 (left, top, right, bottom)
            
        Returns:
            Image.Image: 裁剪后的图像
        """
        left, top, right, bottom = pad_info
        w, h = original_size
        return image.crop((left, top, left + w, top + h))
    
    def _embed_small_image(self, image: Image.Image) -> Image.Image:
        """
        处理小于200x200的图像
        对其周围进行填充使其变成200x200，将原始部分置于200x200中心区域
        
        Args:
            image: 输入图像
            
        Returns:
            Image.Image: 嵌入水印后的图像
        """
        original_size = image.size
        
        # 填充到200x200，原图居中
        padded, pad_info = self._pad_image_center(image)
        
        # 转换为张量
        tensor = self.core.image_to_tensor(padded)
        
        # 嵌入水印
        watermarked_tensor = self.core.embed_watermark_single(tensor, self.secret_pattern)
        
        # 转换回图像
        watermarked = self.core.tensor_to_image(watermarked_tensor)
        
        # 去除填充，恢复原始大小
        result = self._unpad_image(watermarked, original_size, pad_info)
        
        return result
    
    def _embed_large_image(self, image: Image.Image) -> Image.Image:
        """
        处理大于200x200的图像，使用无重叠的滑动窗口（批量推理）
        对于不足200x200的边缘部分，填充后嵌入水印再去除填充
        
        Args:
            image: 输入图像
            
        Returns:
            Image.Image: 嵌入水印后的图像
        """
        w, h = image.size
        window = self.window_size  # 200
        
        # 计算需要的窗口数量（向上取整）
        n_cols = (w + window - 1) // window
        n_rows = (h + window - 1) // window
        
        # 计算填充后的大小
        padded_w = n_cols * window
        padded_h = n_rows * window
        
        # 创建填充后的图像（用于处理边缘不足200的部分）
        padded = Image.new('RGB', (padded_w, padded_h), (128, 128, 128))
        padded.paste(image, (0, 0))
        
        # 转换为numpy数组处理
        padded_array = np.array(padded)
        result_array = padded_array.copy()
        
        # 分批处理窗口，避免显存占用过大
        batch_size = 16  # 每次处理16个窗口
        all_positions = []
        
        for row in range(n_rows):
            for col in range(n_cols):
                x_start = col * window
                y_start = row * window
                x_end = x_start + window
                y_end = y_start + window
                all_positions.append((x_start, y_start, x_end, y_end))
        
        # 分批处理所有窗口
        for batch_idx in range(0, len(all_positions), batch_size):
            batch_positions = all_positions[batch_idx:batch_idx + batch_size]
            
            # 收集当前批次的窗口
            windows = []
            for x_start, y_start, x_end, y_end in batch_positions:
                window_region = padded_array[y_start:y_end, x_start:x_end]
                window_image = Image.fromarray(window_region)
                windows.append(window_image)
            
            # 批量嵌入水印
            batch_tensor = self.core.images_to_batch_tensor(windows)
            watermarked_batch = self.core.embed_watermark_batch(batch_tensor, self.secret_pattern)
            watermarked_windows = self.core.batch_tensor_to_images(watermarked_batch)
            
            # 将处理后的窗口放回
            for i, (x_start, y_start, x_end, y_end) in enumerate(batch_positions):
                result_array[y_start:y_end, x_start:x_end] = np.array(watermarked_windows[i])
            
            # 显式释放GPU内存
            del batch_tensor, watermarked_batch, watermarked_windows, windows
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 裁剪回原始大小（去除填充部分）
        result = Image.fromarray(result_array[:h, :w])
        
        return result
    
    def embed(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        对单张图像嵌入水印
        
        Args:
            image: 图像路径、PIL图像或numpy数组
            
        Returns:
            Image.Image: 嵌入水印后的图像
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')
        
        w, h = image.size
        
        # 根据图像大小选择处理方式
        if w < self.window_size and h < self.window_size:
            # 小于200x200：填充到中心，嵌入后裁剪回原大小
            return self._embed_small_image(image)
        else:
            # 大于等于200x200：使用无重叠滑动窗口
            return self._embed_large_image(image)
    
    def embed_file(self, 
                   input_path: str, 
                   output_path: str) -> bool:
        """
        对单个图像文件嵌入水印并保存
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            bool: 是否成功
        """
        try:
            result = self.embed(input_path)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存结果
            result.save(output_path)
            return True
        except Exception as e:
            print(f"处理文件失败 {input_path}: {e}")
            return False
    
    def embed_folder(self, 
                     input_folder: str, 
                     output_folder: str,
                     recursive: bool = True,
                     show_progress: bool = True) -> dict:
        """
        对文件夹中的所有图像嵌入水印（支持嵌套文件夹）
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            recursive: 是否递归处理子文件夹
            show_progress: 是否显示进度条
            
        Returns:
            dict: 处理结果统计
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        
        # 收集所有图像文件
        if recursive:
            image_files = []
            for fmt in self.SUPPORTED_FORMATS:
                image_files.extend(input_folder.rglob(f'*{fmt}'))
                image_files.extend(input_folder.rglob(f'*{fmt.upper()}'))
        else:
            image_files = []
            for fmt in self.SUPPORTED_FORMATS:
                image_files.extend(input_folder.glob(f'*{fmt}'))
                image_files.extend(input_folder.glob(f'*{fmt.upper()}'))
        
        image_files = list(set(image_files))  # 去重
        
        results = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        iterator = tqdm(image_files, desc="嵌入水印") if show_progress else image_files
        
        for img_path in iterator:
            # 计算相对路径，保持文件夹结构
            rel_path = img_path.relative_to(input_folder)
            output_path = output_folder / rel_path
            
            if self.embed_file(str(img_path), str(output_path)):
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_files'].append(str(img_path))
        
        return results
    
    def embed_batch(self, 
                    input_path: str, 
                    output_path: str,
                    recursive: bool = True,
                    show_progress: bool = True) -> dict:
        """
        统一的批量处理接口，自动判断输入是文件还是文件夹
        
        Args:
            input_path: 输入路径（文件或文件夹）
            output_path: 输出路径
            recursive: 是否递归处理子文件夹（仅对文件夹有效）
            show_progress: 是否显示进度条
            
        Returns:
            dict: 处理结果统计
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            # 单文件处理
            success = self.embed_file(str(input_path), output_path)
            return {
                'total': 1,
                'success': 1 if success else 0,
                'failed': 0 if success else 1,
                'failed_files': [] if success else [str(input_path)]
            }
        elif input_path.is_dir():
            # 文件夹处理
            return self.embed_folder(str(input_path), output_path, recursive, show_progress)
        else:
            raise ValueError(f"无效的输入路径: {input_path}")
