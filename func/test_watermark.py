# -*- coding: utf-8 -*-
"""
水印嵌入提取测试脚本
测试各种正常和处理情况下的水印有效性
包括：正常情况、裁剪、旋转、压缩、拼接等后处理
"""

import os
import sys
import json
import io
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from func.watermark_embedder import WatermarkEmbedder
from func.watermark_extractor import WatermarkExtractor


class ImageProcessor:
    """图像处理器，提供各种后处理操作"""
    
    @staticmethod
    def crop(image: Image.Image, crop_ratio: float = 0.1) -> Image.Image:
        """
        随机裁剪图像（从四周裁剪指定比例）
        
        Args:
            image: 输入图像
            crop_ratio: 裁剪比例（从边缘裁剪的比例，0-1）
            
        Returns:
            Image.Image: 裁剪后的图像（保持原始分辨率，不resize）
        """
        w, h = image.size
        
        # 计算裁剪的像素数
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        
        # 随机选择裁剪起始位置
        max_left = crop_w
        max_top = crop_h
        
        if max_left > 0 and max_top > 0:
            left = np.random.randint(0, max_left + 1)
            top = np.random.randint(0, max_top + 1)
        else:
            left = 0
            top = 0
        
        right = w - (crop_w - left) if crop_w > left else w
        bottom = h - (crop_h - top) if crop_h > top else h
        
        # 确保裁剪区域有效
        if right <= left or bottom <= top:
            return image
        
        return image.crop((left, top, right, bottom))
    
    @staticmethod
    def rotate(image: Image.Image, angle: float, expand: bool = False) -> Image.Image:
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            expand: 是否扩展图像以容纳完整旋转后的图像
            
        Returns:
            Image.Image: 旋转后的图像
        """
        return image.rotate(angle, expand=expand, fillcolor=(128, 128, 128))
    
    @staticmethod
    def compress_jpeg(image: Image.Image, quality: int = 50) -> Image.Image:
        """
        JPEG压缩
        
        Args:
            image: 输入图像
            quality: JPEG质量 (1-100)
            
        Returns:
            Image.Image: 压缩后的图像
        """
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    
    @staticmethod
    def resize_scale(image: Image.Image, scale: float) -> Image.Image:
        """
        缩放图像
        
        Args:
            image: 输入图像
            scale: 缩放比例
            
        Returns:
            Image.Image: 缩放后的图像
        """
        w, h = image.size
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    @staticmethod
    def splice_horizontal(images: List[Image.Image]) -> Image.Image:
        """
        水平拼接多张图像
        
        Args:
            images: 图像列表
            
        Returns:
            Image.Image: 拼接后的图像
        """
        if not images:
            raise ValueError("图像列表不能为空")
        
        # 统一高度
        max_height = max(img.size[1] for img in images)
        resized_images = []
        for img in images:
            w, h = img.size
            if h != max_height:
                new_w = int(w * max_height / h)
                img = img.resize((new_w, max_height), Image.Resampling.LANCZOS)
            resized_images.append(img)
        
        total_width = sum(img.size[0] for img in resized_images)
        result = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for img in resized_images:
            result.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        return result
    
    @staticmethod
    def splice_vertical(images: List[Image.Image]) -> Image.Image:
        """
        垂直拼接多张图像
        
        Args:
            images: 图像列表
            
        Returns:
            Image.Image: 拼接后的图像
        """
        if not images:
            raise ValueError("图像列表不能为空")
        
        # 统一宽度
        max_width = max(img.size[0] for img in images)
        resized_images = []
        for img in images:
            w, h = img.size
            if w != max_width:
                new_h = int(h * max_width / w)
                img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
            resized_images.append(img)
        
        total_height = sum(img.size[1] for img in resized_images)
        result = Image.new('RGB', (max_width, total_height))
        
        y_offset = 0
        for img in resized_images:
            result.paste(img, (0, y_offset))
            y_offset += img.size[1]
        
        return result
    
    @staticmethod
    def blur(image: Image.Image, radius: int = 2) -> Image.Image:
        """
        模糊处理
        
        Args:
            image: 输入图像
            radius: 模糊半径
            
        Returns:
            Image.Image: 模糊后的图像
        """
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def add_noise(image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """
        添加噪声
        
        Args:
            image: 输入图像
            intensity: 噪声强度 (0-1)
            
        Returns:
            Image.Image: 添加噪声后的图像
        """
        img_array = np.array(image).astype(np.float32)
        noise = np.random.randn(*img_array.shape) * intensity * 255
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    @staticmethod
    def copy_move(source_image: Image.Image, target_image: Image.Image, 
                  crop_ratio: float = 0.3, scale: float = 1.0, seed: int = None) -> Image.Image:
        """
        Copy-Move攻击：从源图像随机裁剪+缩放后粘贴到目标图像
        
        Args:
            source_image: 源图像（复制来源）
            target_image: 目标图像（粘贴目标）
            crop_ratio: 从源图像裁剪的比例 (0-1)
            scale: 缩放比例
            seed: 随机种子（可选，用于复现）
            
        Returns:
            Image.Image: Copy-Move后的图像
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 从源图像随机裁剪
        src_w, src_h = source_image.size
        crop_w = int(src_w * crop_ratio)
        crop_h = int(src_h * crop_ratio)
        
        # 随机选择裁剪起始位置
        max_x = src_w - crop_w
        max_y = src_h - crop_h
        crop_x = np.random.randint(0, max(1, max_x))
        crop_y = np.random.randint(0, max(1, max_y))
        
        # 裁剪
        cropped = source_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        
        # 缩放
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        scaled = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 创建目标图像的副本
        result = target_image.copy()
        tgt_w, tgt_h = target_image.size
        
        # 随机选择粘贴位置（确保不超出边界）
        max_paste_x = tgt_w - new_w
        max_paste_y = tgt_h - new_h
        
        if max_paste_x > 0 and max_paste_y > 0:
            paste_x = np.random.randint(0, max_paste_x)
            paste_y = np.random.randint(0, max_paste_y)
            result.paste(scaled, (paste_x, paste_y))
        else:
            # 如果缩放后的图像太大，缩小到适合目标图像
            fit_scale = min(tgt_w / new_w, tgt_h / new_h) * 0.8
            fit_w = int(new_w * fit_scale)
            fit_h = int(new_h * fit_scale)
            scaled = scaled.resize((fit_w, fit_h), Image.Resampling.LANCZOS)
            paste_x = (tgt_w - fit_w) // 2
            paste_y = (tgt_h - fit_h) // 2
            result.paste(scaled, (paste_x, paste_y))
        
        return result


class WatermarkTester:
    """水印测试器，执行全面的水印嵌入提取测试（批量推理）"""
    
    # 批处理大小
    BATCH_SIZE = 8
    
    def __init__(self, 
                 output_base_dir: str,
                 watermark_secret: Optional[str] = None,
                 device: str = 'cuda',
                 batch_size: int = 8,
                 enable_error_correction: bool = True,
                 eval_mode: bool = False):
        """
        初始化测试器
        
        Args:
            output_base_dir: 输出基础目录
            watermark_secret: 水印密钥（如果为None则随机生成）
            device: 运行设备
            batch_size: 批处理大小
            enable_error_correction: 是否启用纠错功能
        """
        self.output_base_dir = Path(output_base_dir)
        self.device = device
        self.batch_size = batch_size
        self.enable_error_correction = enable_error_correction
        self.eval_mode = eval_mode
        
        # 初始化嵌入器和提取器
        self.embedder = WatermarkEmbedder(watermark_secret=watermark_secret, device=device)
        self.secret = self.embedder.get_secret_string()
        self.extractor = WatermarkExtractor(
            expected_secret=self.secret, 
            device=device
        )
        
        # 创建输出目录结构
        self._setup_directories()
        
        # 测试结果
        self.results = {
            'test_time': datetime.now().isoformat(),
            'watermark_secret': self.secret,
            'error_correction_enabled': enable_error_correction,
            'tests': {}
        }
        
        # 图像处理器
        self.processor = ImageProcessor()
    
    def _create_detail_dict(self, extract_result: Dict, **kwargs) -> Dict:
        """
        创建测试详情字典（包含提取结果的所有信息）
        
        Args:
            extract_result: 提取结果字典
            **kwargs: 额外的键值对
            
        Returns:
            Dict: 详情字典
        """
        detail = {
            'extracted_secret': extract_result['secret'],
            'similarity': extract_result.get('similarity', 0),
            'success': extract_result['success'],
            'method': extract_result['method'],
            'preprocessing': extract_result.get('preprocessing'),
            'error_correction_used': extract_result.get('error_correction_used', False),
            'error_correction_type': extract_result.get('error_correction_type'),
            'corrected_bits': extract_result.get('corrected_bits', 0),
            'similarity_before_correction': extract_result.get('similarity_before_correction', 0),
            'similarity_after_correction': extract_result.get('similarity_after_correction', 0)
        }
        detail.update(kwargs)
        return detail
    
    def _setup_directories(self):
        """创建输出目录结构"""
        # 无水印图像目录
        self.wo_wm_dir = self.output_base_dir / 'wo_wm'
        self.wo_wm_dir.mkdir(parents=True, exist_ok=True)
        
        # 有水印图像目录
        self.w_wm_dir = self.output_base_dir / 'w_wm'
        self.w_wm_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建各种处理类型的子目录
        process_types = [
            'original',   # 无后处理
            'crop',       # 裁剪
            'rotate',     # 旋转
            'compress',   # 压缩
            'splice',     # 拼接
            'scale',      # 缩放
            'blur',       # 模糊
            'noise',      # 噪声
            'copy_move',  # Copy-Move攻击
        ]
        
        for proc_type in process_types:
            (self.wo_wm_dir / proc_type).mkdir(parents=True, exist_ok=True)
            (self.w_wm_dir / proc_type).mkdir(parents=True, exist_ok=True)
    
    def _save_image(self, 
                    image: Image.Image, 
                    is_watermarked: bool, 
                    category: str, 
                    sub_category: str,
                    filename: str) -> str:
        """
        保存图像
        
        Args:
            image: 要保存的图像
            is_watermarked: 是否是水印图像
            category: 处理类别
            sub_category: 子类别（不同参数）
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        base_dir = self.w_wm_dir if is_watermarked else self.wo_wm_dir
        if sub_category:
            save_path = base_dir / category / sub_category / filename
        else:
            save_path = base_dir / category / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(save_path))
        return str(save_path)
    
    def test_normal(self, test_images: List[str]) -> Dict:
        """
        测试正常情况下的水印嵌入和提取（批量处理）
        
        Args:
            test_images: 测试图像路径列表
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试正常情况")
        print("=" * 50)
        
        results = {
            'name': 'normal',
            'description': '正常情况下的水印嵌入和提取',
            'total': len(test_images),
            'success': 0,
            'failed': 0,
            'details': []
        }
        
        # 批量处理图像
        for batch_start in range(0, len(test_images), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(test_images))
            batch_paths = test_images[batch_start:batch_end]
            
            # 加载批量图像
            batch_images = []
            batch_names = []
            for img_path in batch_paths:
                img_name = Path(img_path).stem
                batch_names.append(img_name)
                original_image = Image.open(img_path).convert('RGB')
                batch_images.append(original_image)
            
            # 批量嵌入水印
            watermarked_images = []
            for img in batch_images:
                watermarked = self.embedder.embed(img)
                watermarked_images.append(watermarked)
            
            # 批量保存和提取
            for i, (img_path, img_name, original_image, watermarked) in enumerate(
                zip(batch_paths, batch_names, batch_images, watermarked_images)
            ):
                # 保存原始无水印图像
                wo_path = self._save_image(
                    original_image, False, 'original', '', f'{img_name}.png'
                )
                
                # 保存有水印图像
                w_path = self._save_image(
                    watermarked, True, 'original', '', f'{img_name}.png'
                )
                
                # 提取水印
                extract_result = self.extractor.extract(watermarked, enable_error_correction=self.enable_error_correction)
                
                detail = self._create_detail_dict(
                    extract_result,
                    image=img_path,
                    original_path=wo_path,
                    watermarked_path=w_path,
                    expected_secret=self.secret
                )
                
                results['details'].append(detail)
                
                if extract_result['success']:
                    results['success'] += 1
                    print(f"  ✓ {img_name}: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
                else:
                    results['failed'] += 1
                    print(f"  ✗ {img_name}: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['normal'] = results
        return results
    
    def test_crop(self, test_images: List[str]) -> Dict:
        """
        测试裁剪处理后的水印提取（批量处理）
        
        Args:
            test_images: 测试图像路径列表
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试裁剪处理")
        print("=" * 50)
        
        crop_params = [
            {'crop_ratio': 0.10},
            {'crop_ratio': 0.20},
            {'crop_ratio': 0.30},
            {'crop_ratio': 0.40},
            {'crop_ratio': 0.50},
            {'crop_ratio': 0.60},
            {'crop_ratio': 0.70},
            {'crop_ratio': 0.80},
            {'crop_ratio': 0.90},
        ]
        
        results = {
            'name': 'crop',
            'description': '裁剪处理后的水印提取',
            'parameters': crop_params,
            'total': len(test_images) * len(crop_params),
            'success': 0,
            'failed': 0,
            'by_parameter': {},
            'details': []
        }
        
        # 不预加载所有图像，避免内存占用过大
        
        for param in crop_params:
            param_key = f"ratio_{param['crop_ratio']}"
            results['by_parameter'][param_key] = {'success': 0, 'failed': 0}
            
            print(f"\n  裁剪参数: 比例={param['crop_ratio']}")
            
            # 按需处理图像
            for img_path in test_images:
                img_name = Path(img_path).stem
                original_image = Image.open(img_path).convert('RGB')
                watermarked = self.embedder.embed(original_image)
                
                # 对无水印和有水印图像都进行裁剪
                cropped_wo = self.processor.crop(original_image, **param)
                cropped_w = self.processor.crop(watermarked, **param)
                
                # 保存图像
                wo_path = self._save_image(
                    cropped_wo, False, 'crop', param_key, f'{img_name}.png'
                )
                w_path = self._save_image(
                    cropped_w, True, 'crop', param_key, f'{img_name}.png'
                )
                
                # 提取水印
                extract_result = self.extractor.extract(cropped_w, enable_error_correction=self.enable_error_correction)
                
                detail = self._create_detail_dict(
                    extract_result,
                    image=img_path,
                    parameter=param,
                    original_path=wo_path,
                    watermarked_path=w_path
                )
                
                results['details'].append(detail)
                
                if extract_result['success']:
                    results['success'] += 1
                    results['by_parameter'][param_key]['success'] += 1
                    print(f"    ✓ {img_name}: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
                else:
                    results['failed'] += 1
                    results['by_parameter'][param_key]['failed'] += 1
                    print(f"    ✗ {img_name}: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['crop'] = results
        return results
    
    def test_rotate(self, test_images: List[str]) -> Dict:
        """
        测试旋转处理后的水印提取（批量处理）
        
        Args:
            test_images: 测试图像路径列表
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试旋转处理")
        print("=" * 50)
        
        rotate_params = [
            {'angle': 5, 'expand': False},
            {'angle': 10, 'expand': False},
            {'angle': 15, 'expand': False},
            {'angle': 30, 'expand': False},
            {'angle': 45, 'expand': True},
            {'angle': 90, 'expand': False},
            {'angle': 180, 'expand': False},
        ]
        
        results = {
            'name': 'rotate',
            'description': '旋转处理后的水印提取',
            'parameters': rotate_params,
            'total': len(test_images) * len(rotate_params),
            'success': 0,
            'failed': 0,
            'by_parameter': {},
            'details': []
        }
        
        for param in rotate_params:
            param_key = f"angle_{param['angle']}_expand_{param['expand']}"
            results['by_parameter'][param_key] = {'success': 0, 'failed': 0}
            
            print(f"\n  旋转参数: 角度={param['angle']}°, 扩展={param['expand']}")
            
            for img_path in test_images:
                img_name = Path(img_path).stem
                original_image = Image.open(img_path).convert('RGB')
                watermarked = self.embedder.embed(original_image)
                
                # 对无水印和有水印图像都进行旋转
                rotated_wo = self.processor.rotate(original_image, **param)
                rotated_w = self.processor.rotate(watermarked, **param)
                
                # 保存图像
                wo_path = self._save_image(
                    rotated_wo, False, 'rotate', param_key, f'{img_name}.png'
                )
                w_path = self._save_image(
                    rotated_w, True, 'rotate', param_key, f'{img_name}.png'
                )
                
                # 提取水印
                extract_result = self.extractor.extract(rotated_w, enable_error_correction=self.enable_error_correction)
                
                detail = self._create_detail_dict(
                    extract_result,
                    image=img_path,
                    parameter=param,
                    original_path=wo_path,
                    watermarked_path=w_path
                )
                
                results['details'].append(detail)
                
                if extract_result['success']:
                    results['success'] += 1
                    results['by_parameter'][param_key]['success'] += 1
                    print(f"    ✓ {img_name}: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
                else:
                    results['failed'] += 1
                    results['by_parameter'][param_key]['failed'] += 1
                    print(f"    ✗ {img_name}: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['rotate'] = results
        return results
    
    def test_compress(self, test_images: List[str]) -> Dict:
        """
        测试JPEG压缩处理后的水印提取（批量处理）
        
        Args:
            test_images: 测试图像路径列表
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试JPEG压缩处理")
        print("=" * 50)
        
        compress_params = [
            {'quality': 90},
            {'quality': 70},
            {'quality': 50},
            {'quality': 30},
            {'quality': 10},
        ]
        
        results = {
            'name': 'compress',
            'description': 'JPEG压缩处理后的水印提取',
            'parameters': compress_params,
            'total': len(test_images) * len(compress_params),
            'success': 0,
            'failed': 0,
            'by_parameter': {},
            'details': []
        }
        
        for param in compress_params:
            param_key = f"quality_{param['quality']}"
            results['by_parameter'][param_key] = {'success': 0, 'failed': 0}
            
            print(f"\n  压缩参数: 质量={param['quality']}")
            
            for img_path in test_images:
                img_name = Path(img_path).stem
                original_image = Image.open(img_path).convert('RGB')
                watermarked = self.embedder.embed(original_image)
                
                # 对无水印和有水印图像都进行压缩
                compressed_wo = self.processor.compress_jpeg(original_image, **param)
                compressed_w = self.processor.compress_jpeg(watermarked, **param)
                
                # 保存图像
                wo_path = self._save_image(
                    compressed_wo, False, 'compress', param_key, f'{img_name}.jpg'
                )
                w_path = self._save_image(
                    compressed_w, True, 'compress', param_key, f'{img_name}.jpg'
                )
                
                # 提取水印
                extract_result = self.extractor.extract(compressed_w, enable_error_correction=self.enable_error_correction)
                
                detail = self._create_detail_dict(
                    extract_result,
                    image=img_path,
                    parameter=param,
                    original_path=wo_path,
                    watermarked_path=w_path
                )
                
                results['details'].append(detail)
                
                if extract_result['success']:
                    results['success'] += 1
                    results['by_parameter'][param_key]['success'] += 1
                    print(f"    ✓ {img_name}: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
                else:
                    results['failed'] += 1
                    results['by_parameter'][param_key]['failed'] += 1
                    print(f"    ✗ {img_name}: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['compress'] = results
        return results
    
    def test_scale(self, test_images: List[str]) -> Dict:
        """
        测试缩放处理后的水印提取（批量处理）
        
        Args:
            test_images: 测试图像路径列表
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试缩放处理")
        print("=" * 50)
        
        scale_params = [
            {'scale': 0.1},
            {'scale': 0.3},
            {'scale': 0.5},
            {'scale': 0.7},
            {'scale': 0.9},
            {'scale': 1.2},
            {'scale': 1.5},
            {'scale': 1.8},
            {'scale': 2.0},
        ]
        
        results = {
            'name': 'scale',
            'description': '缩放处理后的水印提取',
            'parameters': scale_params,
            'total': len(test_images) * len(scale_params),
            'success': 0,
            'failed': 0,
            'by_parameter': {},
            'details': []
        }
        
        for param in scale_params:
            param_key = f"scale_{param['scale']}"
            results['by_parameter'][param_key] = {'success': 0, 'failed': 0}
            
            print(f"\n  缩放参数: 比例={param['scale']}")
            
            for img_path in test_images:
                img_name = Path(img_path).stem
                original_image = Image.open(img_path).convert('RGB')
                watermarked = self.embedder.embed(original_image)
                
                # 对无水印和有水印图像都进行缩放
                scaled_wo = self.processor.resize_scale(original_image, **param)
                scaled_w = self.processor.resize_scale(watermarked, **param)
                
                # 保存图像
                wo_path = self._save_image(
                    scaled_wo, False, 'scale', param_key, f'{img_name}.png'
                )
                w_path = self._save_image(
                    scaled_w, True, 'scale', param_key, f'{img_name}.png'
                )
                
                # 提取水印
                extract_result = self.extractor.extract(scaled_w, enable_error_correction=self.enable_error_correction)
                
                detail = self._create_detail_dict(
                    extract_result,
                    image=img_path,
                    parameter=param,
                    original_path=wo_path,
                    watermarked_path=w_path
                )
                
                results['details'].append(detail)
                
                if extract_result['success']:
                    results['success'] += 1
                    results['by_parameter'][param_key]['success'] += 1
                    print(f"    ✓ {img_name}: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
                else:
                    results['failed'] += 1
                    results['by_parameter'][param_key]['failed'] += 1
                    print(f"    ✗ {img_name}: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['scale'] = results
        return results
    
    def test_blur(self, test_images: List[str]) -> Dict:
        """
        测试模糊处理后的水印提取（批量处理）
        
        Args:
            test_images: 测试图像路径列表
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试模糊处理")
        print("=" * 50)
        
        blur_params = [
            {'radius': 1},
            {'radius': 2},
            {'radius': 3},
            {'radius': 5},
        ]
        
        results = {
            'name': 'blur',
            'description': '模糊处理后的水印提取',
            'parameters': blur_params,
            'total': len(test_images) * len(blur_params),
            'success': 0,
            'failed': 0,
            'by_parameter': {},
            'details': []
        }
        
        for param in blur_params:
            param_key = f"radius_{param['radius']}"
            results['by_parameter'][param_key] = {'success': 0, 'failed': 0}
            
            print(f"\n  模糊参数: 半径={param['radius']}")
            
            for img_path in test_images:
                img_name = Path(img_path).stem
                original_image = Image.open(img_path).convert('RGB')
                watermarked = self.embedder.embed(original_image)
                
                # 对无水印和有水印图像都进行模糊
                blurred_wo = self.processor.blur(original_image, **param)
                blurred_w = self.processor.blur(watermarked, **param)
                
                # 保存图像
                wo_path = self._save_image(
                    blurred_wo, False, 'blur', param_key, f'{img_name}.png'
                )
                w_path = self._save_image(
                    blurred_w, True, 'blur', param_key, f'{img_name}.png'
                )
                
                # 提取水印
                extract_result = self.extractor.extract(blurred_w, enable_error_correction=self.enable_error_correction)
                
                detail = self._create_detail_dict(
                    extract_result,
                    image=img_path,
                    parameter=param,
                    original_path=wo_path,
                    watermarked_path=w_path
                )
                
                results['details'].append(detail)
                
                if extract_result['success']:
                    results['success'] += 1
                    results['by_parameter'][param_key]['success'] += 1
                    print(f"    ✓ {img_name}: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
                else:
                    results['failed'] += 1
                    results['by_parameter'][param_key]['failed'] += 1
                    print(f"    ✗ {img_name}: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['blur'] = results
        return results
    
    def test_noise(self, test_images: List[str]) -> Dict:
        """
        测试噪声处理后的水印提取（批量处理）
        
        Args:
            test_images: 测试图像路径列表
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试噪声处理")
        print("=" * 50)
        
        noise_params = [
            {'intensity': 0.05},
            {'intensity': 0.10},
            {'intensity': 0.15},
            {'intensity': 0.20},
        ]
        
        results = {
            'name': 'noise',
            'description': '噪声处理后的水印提取',
            'parameters': noise_params,
            'total': len(test_images) * len(noise_params),
            'success': 0,
            'failed': 0,
            'by_parameter': {},
            'details': []
        }
        
        for param in noise_params:
            param_key = f"intensity_{param['intensity']}"
            results['by_parameter'][param_key] = {'success': 0, 'failed': 0}
            
            print(f"\n  噪声参数: 强度={param['intensity']}")
            
            for img_path in test_images:
                img_name = Path(img_path).stem
                original_image = Image.open(img_path).convert('RGB')
                watermarked = self.embedder.embed(original_image)
                
                # 对无水印和有水印图像都添加噪声
                noisy_wo = self.processor.add_noise(original_image, **param)
                noisy_w = self.processor.add_noise(watermarked, **param)
                
                # 保存图像
                wo_path = self._save_image(
                    noisy_wo, False, 'noise', param_key, f'{img_name}.png'
                )
                w_path = self._save_image(
                    noisy_w, True, 'noise', param_key, f'{img_name}.png'
                )
                
                # 提取水印
                extract_result = self.extractor.extract(noisy_w, enable_error_correction=self.enable_error_correction)
                
                detail = self._create_detail_dict(
                    extract_result,
                    image=img_path,
                    parameter=param,
                    original_path=wo_path,
                    watermarked_path=w_path
                )
                
                results['details'].append(detail)
                
                if extract_result['success']:
                    results['success'] += 1
                    results['by_parameter'][param_key]['success'] += 1
                    print(f"    ✓ {img_name}: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
                else:
                    results['failed'] += 1
                    results['by_parameter'][param_key]['failed'] += 1
                    print(f"    ✗ {img_name}: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['noise'] = results
        return results
    
    def test_splice(self, test_images: List[str]) -> Dict:
        """
        测试拼接处理后的水印提取
        
        Args:
            test_images: 测试图像路径列表（至少需要2张图像）
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试拼接处理")
        print("=" * 50)
        
        if len(test_images) < 2:
            print("  警告: 拼接测试需要至少2张图像，跳过此测试")
            return {'name': 'splice', 'skipped': True, 'reason': '图像数量不足'}
        
        results = {
            'name': 'splice',
            'description': '拼接处理后的水印提取',
            'total': 0,
            'success': 0,
            'failed': 0,
            'by_parameter': {
                'horizontal': {'success': 0, 'failed': 0},
                'vertical': {'success': 0, 'failed': 0}
            },
            'details': []
        }
        
        # 测试水平拼接
        print("\n  水平拼接测试:")
        images_wo = [Image.open(p).convert('RGB') for p in test_images[:2]]
        images_w = [self.embedder.embed(img) for img in images_wo]
        
        spliced_wo = self.processor.splice_horizontal(images_wo)
        spliced_w = self.processor.splice_horizontal(images_w)
        
        wo_path = self._save_image(
            spliced_wo, False, 'splice', 'horizontal', 'spliced.png'
        )
        w_path = self._save_image(
            spliced_w, True, 'splice', 'horizontal', 'spliced.png'
        )
        
        extract_result = self.extractor.extract(spliced_w, enable_error_correction=self.enable_error_correction)
        results['total'] += 1
        
        detail = self._create_detail_dict(
            extract_result,
            splice_type='horizontal',
            source_images=test_images[:2],
            original_path=wo_path,
            watermarked_path=w_path
        )
        results['details'].append(detail)
        
        if extract_result['success']:
            results['success'] += 1
            results['by_parameter']['horizontal']['success'] += 1
            print(f"    ✓ 水平拼接: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
        else:
            results['failed'] += 1
            results['by_parameter']['horizontal']['failed'] += 1
            print(f"    ✗ 水平拼接: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        # 测试垂直拼接
        print("\n  垂直拼接测试:")
        spliced_wo = self.processor.splice_vertical(images_wo)
        spliced_w = self.processor.splice_vertical(images_w)
        
        wo_path = self._save_image(
            spliced_wo, False, 'splice', 'vertical', 'spliced.png'
        )
        w_path = self._save_image(
            spliced_w, True, 'splice', 'vertical', 'spliced.png'
        )
        
        extract_result = self.extractor.extract(spliced_w, enable_error_correction=self.enable_error_correction)
        results['total'] += 1
        
        detail = self._create_detail_dict(
            extract_result,
            splice_type='vertical',
            source_images=test_images[:2],
            original_path=wo_path,
            watermarked_path=w_path
        )
        results['details'].append(detail)
        
        if extract_result['success']:
            results['success'] += 1
            results['by_parameter']['vertical']['success'] += 1
            print(f"    ✓ 垂直拼接: 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
        else:
            results['failed'] += 1
            results['by_parameter']['vertical']['failed'] += 1
            print(f"    ✗ 垂直拼接: 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['splice'] = results
        return results
    
    def test_copy_move(self, test_images: List[str]) -> Dict:
        """
        测试Copy-Move攻击后的水印提取
        
        Args:
            test_images: 测试图像路径列表（至少需要2张图像）
            
        Returns:
            Dict: 测试结果
        """
        print("\n" + "=" * 50)
        print("测试Copy-Move攻击")
        print("=" * 50)
        
        if len(test_images) < 2:
            print("  警告: Copy-Move测试需要至少2张图像，跳过此测试")
            return {'name': 'copy_move', 'skipped': True, 'reason': '图像数量不足'}
        
        copy_move_params = [
            {'crop_ratio': 0.1, 'scale': 0.5},
            {'crop_ratio': 0.2, 'scale': 0.8},
            {'crop_ratio': 0.2, 'scale': 1.2},
            {'crop_ratio': 0.3, 'scale': 0.7},
            {'crop_ratio': 0.3, 'scale': 1.0},
            {'crop_ratio': 0.3, 'scale': 1.5},
            {'crop_ratio': 0.4, 'scale': 0.6},
            {'crop_ratio': 0.4, 'scale': 1.0},
            {'crop_ratio': 0.5, 'scale': 0.5},
            {'crop_ratio': 0.5, 'scale': 0.9},
        ]
        
        results = {
            'name': 'copy_move',
            'description': 'Copy-Move攻击后的水印提取',
            'parameters': copy_move_params,
            'total': 0,
            'success': 0,
            'failed': 0,
            'by_parameter': {},
            'details': []
        }
        
        # 初始化参数统计
        for param in copy_move_params:
            param_key = f"crop_{param['crop_ratio']}_scale_{param['scale']}"
            results['by_parameter'][param_key] = {'success': 0, 'failed': 0}
        
        # 对每对图像进行测试
        for i in range(len(test_images) - 1):
            source_path = test_images[i]
            target_path = test_images[i + 1]
            
            source_name = Path(source_path).stem
            target_name = Path(target_path).stem
            
            for param in copy_move_params:
                param_key = f"crop_{param['crop_ratio']}_scale_{param['scale']}"
                print(f"\n  Copy-Move参数: 源={source_name}, 目标={target_name}, "
                      f"裁剪比例={param['crop_ratio']}, 缩放={param['scale']}")
                
                # 加载图像
                source_img = Image.open(source_path).convert('RGB')
                target_img = Image.open(target_path).convert('RGB')
                
                # 嵌入水印（在源图像上）
                watermarked_source = self.embedder.embed(source_img)
                
                # 对无水印和有水印图像都进行Copy-Move
                # 从源图像裁剪+缩放后粘贴到目标图像
                cm_wo = self.processor.copy_move(source_img, target_img, **param, seed=42)
                cm_w = self.processor.copy_move(watermarked_source, target_img, **param, seed=42)
                
                # 保存图像
                filename = f'{source_name}_to_{target_name}.png'
                wo_path = self._save_image(
                    cm_wo, False, 'copy_move', param_key, filename
                )
                w_path = self._save_image(
                    cm_w, True, 'copy_move', param_key, filename
                )
                
                # 提取水印
                extract_result = self.extractor.extract(cm_w, enable_error_correction=self.enable_error_correction)
                results['total'] += 1
                
                detail = self._create_detail_dict(
                    extract_result,
                    source_image=source_path,
                    target_image=target_path,
                    parameter=param,
                    original_path=wo_path,
                    watermarked_path=w_path
                )
                results['details'].append(detail)
                
                if extract_result['success']:
                    results['success'] += 1
                    results['by_parameter'][param_key]['success'] += 1
                    print(f"    ✓ 成功 (相似度: {extract_result.get('similarity', 1.0):.2%})")
                else:
                    results['failed'] += 1
                    results['by_parameter'][param_key]['failed'] += 1
                    print(f"    ✗ 失败 (相似度: {extract_result.get('similarity', 0):.2%})")
        
        self.results['tests']['copy_move'] = results
        return results
    
    def run_all_tests(self, test_images: List[str]) -> Dict:
        """
        运行所有测试
        
        Args:
            test_images: 测试图像路径列表
            
        Returns:
            Dict: 所有测试结果
        """
        print("\n" + "=" * 60)
        print("开始水印嵌入提取全面测试")
        print(f"测试图像数量: {len(test_images)}")
        print(f"水印密钥: {self.secret}")
        print("=" * 60)
        
        # 保存测试图像列表
        self.results['test_images'] = test_images
        
        # 运行各项测试
        self.test_normal(test_images)
        if self.eval_mode:
            self.test_crop(test_images)
            self.test_rotate(test_images)
            self.test_compress(test_images)
            self.test_scale(test_images)
            self.test_blur(test_images)
            self.test_noise(test_images)
            self.test_splice(test_images)
            self.test_copy_move(test_images)
        
        # 汇总结果
        self._summarize_results(test_images)
        
        return self.results
    
    def _summarize_results(self, test_images: List[str]):
        """汇总测试结果"""
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        total_tests = 0
        total_success = 0
        total_failed = 0
        
        # ========== 按处理类型和参数的平均结果汇总 ==========
        summary_by_process = []  # 所有处理类型的平均结果列表
        
        for test_name, test_result in self.results['tests'].items():
            if test_result.get('skipped'):
                print(f"  {test_name}: 跳过 - {test_result.get('reason', '未知原因')}")
                summary_by_process.append({
                    'process_type': test_name,
                    'skipped': True,
                    'reason': test_result.get('reason', '未知原因')
                })
                continue
            
            total_tests += test_result.get('total', 0)
            total_success += test_result.get('success', 0)
            total_failed += test_result.get('failed', 0)
            
            test_total = test_result.get('total', 0)
            test_success = test_result.get('success', 0)
            success_rate = test_success / max(test_total, 1)
            
            # 计算该处理类型的平均相似度
            similarities = [d.get('similarity', 0) for d in test_result.get('details', [])]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            process_summary = {
                'process_type': test_name,
                'description': test_result.get('description', ''),
                'total': test_total,
                'success': test_success,
                'failed': test_result.get('failed', 0),
                'success_rate': success_rate,
                'avg_similarity': avg_similarity,
            }
            
            # 如果有按参数的统计，添加参数级别的汇总
            if 'by_parameter' in test_result:
                process_summary['by_parameter'] = {}
                for param_key, param_stats in test_result['by_parameter'].items():
                    param_total = param_stats.get('success', 0) + param_stats.get('failed', 0)
                    param_success = param_stats.get('success', 0)
                    param_rate = param_success / max(param_total, 1)
                    
                    # 计算该参数的平均相似度
                    param_similarities = [
                        d.get('similarity', 0) for d in test_result.get('details', [])
                        if self._match_param_key(d, param_key, test_name)
                    ]
                    param_avg_similarity = sum(param_similarities) / len(param_similarities) if param_similarities else 0
                    
                    process_summary['by_parameter'][param_key] = {
                        'total': param_total,
                        'success': param_success,
                        'failed': param_stats.get('failed', 0),
                        'success_rate': param_rate,
                        'avg_similarity': param_avg_similarity
                    }
            
            summary_by_process.append(process_summary)
            print(f"  {test_name}: {test_success}/{test_total} 成功 ({success_rate:.2%}), 平均相似度: {avg_similarity:.2%}")
        
        overall_rate = total_success / max(total_tests, 1)
        print(f"\n  总计: {total_success}/{total_tests} 成功 ({overall_rate:.2%})")
        
        # ========== 按每个图像的结果汇总 ==========
        summary_by_image = []  # 每个图像的结果列表
        
        for img_path in test_images:
            img_name = Path(img_path).stem
            image_results = {
                'image_path': img_path,
                'image_name': img_name,
                'tests': {}
            }
            
            # 遍历所有测试结果，收集该图像的数据
            for test_name, test_result in self.results['tests'].items():
                if test_result.get('skipped'):
                    continue
                
                # 收集该图像在该测试中的所有结果
                img_details = []
                for detail in test_result.get('details', []):
                    # 检查是否是该图像的结果
                    detail_img = detail.get('image', '')
                    if detail_img == img_path or Path(detail_img).stem == img_name:
                        img_details.append(detail)
                    # 拼接测试中可能包含多个图像
                    source_images = detail.get('source_images', [])
                    if img_path in source_images:
                        img_details.append(detail)
                
                if img_details:
                    successes = sum(1 for d in img_details if d.get('success', False))
                    total = len(img_details)
                    similarities = [d.get('similarity', 0) for d in img_details]
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                    
                    image_results['tests'][test_name] = {
                        'total': total,
                        'success': successes,
                        'failed': total - successes,
                        'success_rate': successes / max(total, 1),
                        'avg_similarity': avg_similarity,
                        'details': img_details
                    }
            
            # 计算该图像的整体统计
            all_successes = sum(t['success'] for t in image_results['tests'].values())
            all_total = sum(t['total'] for t in image_results['tests'].values())
            all_similarities = []
            for t in image_results['tests'].values():
                for d in t.get('details', []):
                    all_similarities.append(d.get('similarity', 0))
            
            image_results['overall'] = {
                'total': all_total,
                'success': all_successes,
                'failed': all_total - all_successes,
                'success_rate': all_successes / max(all_total, 1),
                'avg_similarity': sum(all_similarities) / len(all_similarities) if all_similarities else 0
            }
            
            summary_by_image.append(image_results)
        
        # ========== 保存汇总结果 ==========
        self.results['summary'] = {
            'total_tests': total_tests,
            'total_success': total_success,
            'total_failed': total_failed,
            'success_rate': overall_rate
        }
        
        self.results['summary_by_process'] = summary_by_process
        self.results['summary_by_image'] = summary_by_image
    
    def _match_param_key(self, detail: Dict, param_key: str, test_name: str) -> bool:
        """检查detail是否匹配指定的参数key"""
        param = detail.get('parameter', {})
        
        if test_name == 'crop':
            key = f"ratio_{param.get('crop_ratio')}_pos_{param.get('position')}"
            return key == param_key
        elif test_name == 'rotate':
            key = f"angle_{param.get('angle')}_expand_{param.get('expand')}"
            return key == param_key
        elif test_name == 'compress':
            key = f"quality_{param.get('quality')}"
            return key == param_key
        elif test_name == 'scale':
            key = f"scale_{param.get('scale')}"
            return key == param_key
        elif test_name == 'blur':
            key = f"radius_{param.get('radius')}"
            return key == param_key
        elif test_name == 'noise':
            key = f"intensity_{param.get('intensity')}"
            return key == param_key
        elif test_name == 'splice':
            return detail.get('splice_type') == param_key
        
        return False
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        保存测试结果到JSON文件
        
        Args:
            output_path: 输出文件路径，如果为None则保存到output_base_dir下
            
        Returns:
            str: 保存的文件路径
        """
        if output_path is None:
            output_path = str(self.output_base_dir / 'test_results.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n测试结果已保存到: {output_path}")
        return output_path


def find_test_images(image_dir: str) -> List[str]:
    """
    在目录中查找所有测试图像（不限制数量）
    
    Args:
        image_dir: 图像目录
        
    Returns:
        List[str]: 图像路径列表
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_dir = Path(image_dir)
    
    images = []
    for ext in supported_formats:
        images.extend(image_dir.glob(f'*{ext}'))
        images.extend(image_dir.glob(f'*{ext.upper()}'))
    
    images = list(set(images))
    return [str(p) for p in sorted(images)]


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='水印嵌入提取测试')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='测试图像路径或目录')
    parser.add_argument('--output', '-o', type=str, default='./img/results',
                       help='输出目录 (默认: ./img/results)')
    parser.add_argument('--secret', '-s', type=str, default=None,
                       help='30位二进制水印密钥 (默认: 随机生成)')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       help='运行设备 (默认: cuda)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='批处理大小 (默认: 8)')
    parser.add_argument('--eval-mode', '-e', action='store_true',
                        help='后处理评估')
    
    args = parser.parse_args()
    
    # 查找测试图像（不限制数量）
    input_path = Path(args.input)
    if input_path.is_file():
        test_images = [str(input_path)]
    elif input_path.is_dir():
        test_images = find_test_images(str(input_path))
    else:
        print(f"错误: 无效的输入路径 {args.input}")
        return
    
    if not test_images:
        print(f"错误: 在 {args.input} 中未找到测试图像")
        return
    
    print(f"找到 {len(test_images)} 张测试图像")
    for i, img in enumerate(test_images, 1):
        print(f"  {i}. {Path(img).name}")
    
    # 初始化测试器并运行测试
    tester = WatermarkTester(
        output_base_dir=args.output,
        watermark_secret=args.secret,
        device=args.device,
        batch_size=args.batch_size,
        eval_mode=args.eval_mode

    )
    
    tester.run_all_tests(test_images)
    tester.save_results()


if __name__ == '__main__':
    main()
