# -*- coding: utf-8 -*-
"""
水印提取模块 - 提供图像水印提取功能
支持单图像、文件夹批量处理，多尺度滑动窗口提取和投票机制
包含图像预处理增强功能（JPEG去压缩、非局部均值去噪）
"""

import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Union, List, Tuple, Dict
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import cv2

from .watermark_core import WatermarkCore


class WatermarkExtractor:
    """水印提取器，支持多尺度滑动窗口和投票机制"""
    
    # 支持的图像格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 滑动窗口大小序列
    WINDOW_SIZES = [200, 150, 100, 50]
    
    # 重叠比例
    OVERLAP_RATIO = 0.5
    
    def __init__(self, 
                 expected_secret: Optional[str] = None,
                 similarity_threshold: float = 0.9,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化水印提取器
        
        Args:
            expected_secret: 期望的30位二进制水印密钥（用于验证）
            similarity_threshold: 相似度阈值，默认0.9
            device: 运行设备
        """
        self.core = WatermarkCore(device=device)
        self.device = device
        self.expected_secret = expected_secret
        self.similarity_threshold = similarity_threshold
        self.target_size = WatermarkCore.SUPPORTED_SIZE  # 200
    
    def set_expected_secret(self, expected_secret: str):
        """设置期望的水印密钥"""
        self.expected_secret = expected_secret
    
    def _preprocess_jpeg_enhancement(self, image: Image.Image) -> Image.Image:
        """
        JPEG伪去压缩增强
        尝试减少JPEG压缩伪影，提升水印提取成功率
        
        Args:
            image: 输入图像
            
        Returns:
            Image.Image: 增强后的图像
        """
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 使用双边滤波去除压缩伪影（保留边缘）
            enhanced = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
            
            # 轻微锐化以恢复细节
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1.0
            sharpened = cv2.filter2D(enhanced, -1, kernel * 0.3 + np.eye(3).flatten().reshape(3,3) * 0.7)
            
            return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
        except Exception as e:
            # 如果OpenCV不可用，使用PIL的简单替代方法
            enhanced = image.filter(ImageFilter.SMOOTH_MORE)
            enhancer = ImageEnhance.Sharpness(enhanced)
            return enhancer.enhance(1.2)
    
    def _preprocess_nlmeans_denoise(self, image: Image.Image) -> Image.Image:
        """
        自适应非局部均值去噪（NL-Means）
        去除图像噪声，提升水印提取成功率
        
        Args:
            image: 输入图像
            
        Returns:
            Image.Image: 去噪后的图像
        """
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 使用非局部均值去噪
            denoised = cv2.fastNlMeansDenoisingColored(
                img_array,
                None,
                h=10,                # 滤波强度
                hColor=10,           # 颜色分量滤波强度
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            return Image.fromarray(denoised)
        except Exception as e:
            # 如果OpenCV不可用，使用PIL的简单替代方法
            return image.filter(ImageFilter.MedianFilter(size=3))
    
    def _preprocess_combined(self, image: Image.Image) -> Image.Image:
        """
        组合预处理（JPEG增强 + NL-Means去噪）
        
        Args:
            image: 输入图像
            
        Returns:
            Image.Image: 预处理后的图像
        """
        # 先进行JPEG增强
        enhanced = self._preprocess_jpeg_enhancement(image)
        # 再进行去噪
        denoised = self._preprocess_nlmeans_denoise(enhanced)
        return denoised
    
    def _resize_and_extract(self, image: Image.Image) -> Tuple[str, float]:
        """
        将图像resize到200x200后提取水印
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[str, float]: (提取的密钥字符串, 相似度)
        """
        # Resize到200x200
        resized = image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        # 转换为张量
        tensor = self.core.image_to_tensor(resized)
        
        # 提取水印
        _, secret_str = self.core.extract_secret_single(tensor)
        
        # 计算相似度
        if self.expected_secret:
            is_match, similarity = self.core.compare_secrets(
                self.expected_secret, secret_str, self.similarity_threshold
            )
            return secret_str, similarity
        
        return secret_str, 1.0
    
    def _pad_and_extract(self, image: Image.Image) -> Tuple[str, float]:
        """
        将图像填充到200x200后提取水印（用于小图像）
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[str, float]: (提取的密钥字符串, 相似度)
        """
        w, h = image.size
        
        # 创建200x200的黑色背景
        padded = Image.new('RGB', (self.target_size, self.target_size), color=(0, 0, 0))
        
        # 计算居中位置
        paste_x = (self.target_size - w) // 2
        paste_y = (self.target_size - h) // 2
        
        # 将原图粘贴到中心
        padded.paste(image, (paste_x, paste_y))
        
        # 转换为张量
        tensor = self.core.image_to_tensor(padded)
        
        # 提取水印
        _, secret_str = self.core.extract_secret_single(tensor)
        
        # 计算相似度
        if self.expected_secret:
            is_match, similarity = self.core.compare_secrets(
                self.expected_secret, secret_str, self.similarity_threshold
            )
            return secret_str, similarity
        
        return secret_str, 1.0
    
    def _extract_window(self, image: Image.Image, window_size: int) -> List[Tuple[str, float]]:
        """
        使用指定大小的滑动窗口提取水印（50%重叠）（批量推理）
        
        Args:
            image: 输入图像
            window_size: 窗口大小
            
        Returns:
            List[Tuple[str, float]]: 提取到的所有(水印密钥, 相似度)列表
        """
        w, h = image.size
        
        # 如果图像比窗口小，直接resize提取
        if w < window_size or h < window_size:
            secret, similarity = self._resize_and_extract(image)
            return [(secret, similarity)]
        
        # 计算步长（50%重叠）
        stride = int(window_size * (1 - self.OVERLAP_RATIO))
        
        image_array = np.array(image)
        
        # 收集所有窗口位置
        window_positions = []
        y = 0
        while y + window_size <= h:
            x = 0
            while x + window_size <= w:
                window_positions.append((x, y))
                x += stride
            y += stride
        
        if not window_positions:
            return []
        
        # 分批处理窗口，避免显存占用过大
        batch_size = 16
        extracted_results = []
        
        for batch_idx in range(0, len(window_positions), batch_size):
            batch_positions = window_positions[batch_idx:batch_idx + batch_size]
            
            # 收集当前批次的窗口
            windows = []
            for x, y in batch_positions:
                window_region = image_array[y:y+window_size, x:x+window_size]
                window_image = Image.fromarray(window_region)
                
                # Resize到200x200（如果窗口不是200x200）
                if window_size != self.target_size:
                    resized_window = window_image.resize(
                        (self.target_size, self.target_size), 
                        Image.Resampling.LANCZOS
                    )
                else:
                    resized_window = window_image
                
                windows.append(resized_window)
            
            # 批量提取水印
            batch_tensor = self.core.images_to_batch_tensor(windows)
            _, secret_strs = self.core.extract_secret_batch(batch_tensor)
            
            # 计算相似度
            for secret_str in secret_strs:
                if self.expected_secret:
                    _, similarity = self.core.compare_secrets(
                        self.expected_secret, secret_str, self.similarity_threshold
                    )
                else:
                    similarity = 1.0
                extracted_results.append((secret_str, similarity))
            
            # 显式释放GPU内存
            del batch_tensor, secret_strs, windows
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return extracted_results
    
    def _vote_secret(self, results: List[Tuple[str, float]]) -> Tuple[Optional[str], float, float]:
        """
        对多个提取到的水印密钥进行投票
        
        Args:
            results: (水印密钥, 相似度)列表
            
        Returns:
            Tuple[Optional[str], float, float]: (得票最多的密钥, 最高相似度, 投票比例)
        """
        if not results:
            return None, 0.0, 0.0
        
        secrets = [r[0] for r in results]
        
        # 使用Counter进行投票
        counter = Counter(secrets)
        most_common = counter.most_common(1)[0]
        
        voted_secret = most_common[0]
        vote_count = most_common[1]
        vote_ratio = vote_count / len(secrets)
        
        # 找到该密钥对应的最高相似度
        max_similarity = max(r[1] for r in results if r[0] == voted_secret)
        
        return voted_secret, max_similarity, vote_ratio
    
    def extract(self, 
                image: Union[str, Image.Image, np.ndarray],
                use_multiscale: bool = True,
                enable_error_correction: bool = True,
                max_hamming_corrections: int = 3) -> Dict:
        """
        从单张图像提取水印（支持纠错）
        
        提取流程：
        a. 先对图像resize到200x200进行水印提取，如果提取到水印则成功
        b. 保持图像大小不变，依次以[200x200, 150x150, 100x100, 50x50]有重叠的滑动窗口进行提取
        c. 一张图像可能提取到多个不同水印，采用投票形式决定最终水印
        d. 当前大小滑动窗口提取到水印则不进行下个大小窗口提取
        e. 如果启用纠错，对未通过阈值的水印尝试纠错
        
        Args:
            image: 图像路径、PIL图像或numpy数组
            use_multiscale: 是否使用多尺度滑动窗口
            enable_error_correction: 是否启用纠错功能
            max_hamming_corrections: 汉明纠错最大允许纠正的比特数（默认3比特）
            
        Returns:
            Dict: 提取结果
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')
        
        result = {
            'success': False,
            'secret': None,
            'similarity': 0.0,
            'method': None,
            'vote_ratio': 0.0,
            'window_size': None,
            'preprocessing': None,  # 预处理方法
            # 纠错相关信息
            'error_correction_used': False,
            'error_correction_type': None,
            'corrected_bits': 0,
            'similarity_before_correction': 0.0,
            'similarity_after_correction': 0.0
        }
        
        # 步骤a: 先对图像resize到200x200进行水印提取
        secret, similarity = self._resize_and_extract(image)
        
        # 记录纠错前的相似度
        original_secret = secret
        original_similarity = similarity
        
        # 尝试汉明纠错（如果启用且有期望密钥）
        if enable_error_correction and self.expected_secret and similarity < self.similarity_threshold:
            corrected_secret, corrected_bits, did_correction = self.core.error_correction_hamming(
                secret, self.expected_secret, max_hamming_corrections
            )
            if did_correction:
                secret = corrected_secret
                _, similarity = self.core.compare_secrets(
                    self.expected_secret, secret, self.similarity_threshold
                )
                result['error_correction_used'] = True
                result['error_correction_type'] = 'hamming'
                result['corrected_bits'] = corrected_bits
                result['similarity_before_correction'] = original_similarity
                result['similarity_after_correction'] = similarity
        
        if similarity >= self.similarity_threshold:
            result['success'] = True
            result['secret'] = secret
            result['similarity'] = similarity
            result['method'] = 'resize'
            return result
        
        # 对于小于200x200的图像，尝试填充方式提取
        w, h = image.size
        if w < self.target_size or h < self.target_size:
            pad_secret, pad_similarity = self._pad_and_extract(image)
            
            # 尝试汉明纠错（如果启用且有期望密钥）
            if enable_error_correction and self.expected_secret and pad_similarity < self.similarity_threshold:
                corrected_secret, corrected_bits, did_correction = self.core.error_correction_hamming(
                    pad_secret, self.expected_secret, max_hamming_corrections
                )
                if did_correction:
                    pad_secret = corrected_secret
                    _, pad_similarity = self.core.compare_secrets(
                        self.expected_secret, pad_secret, self.similarity_threshold
                    )
                    if pad_similarity >= self.similarity_threshold:
                        result['success'] = True
                        result['secret'] = pad_secret
                        result['similarity'] = pad_similarity
                        result['method'] = 'pad'
                        result['error_correction_used'] = True
                        result['error_correction_type'] = 'hamming'
                        result['corrected_bits'] = corrected_bits
                        result['similarity_before_correction'] = original_similarity
                        result['similarity_after_correction'] = pad_similarity
                        return result
            
            if pad_similarity >= self.similarity_threshold:
                result['success'] = True
                result['secret'] = pad_secret
                result['similarity'] = pad_similarity
                result['method'] = 'pad'
                return result
        
        if not use_multiscale:
            # 如果不使用多尺度，直接返回resize提取的结果
            result['secret'] = secret
            result['similarity'] = similarity
            result['method'] = 'resize'
            return result
        
        # 步骤b: 使用多尺度滑动窗口
        for window_size in self.WINDOW_SIZES:
            extracted_results = self._extract_window(image, window_size)
            
            if not extracted_results:
                continue
            
            # 尝试投票纠错（如果启用）
            all_secrets = [s for s, _ in extracted_results]
            if enable_error_correction and len(all_secrets) > 1:
                voted_secret, corrections_count, voting_details = self.core.error_correction_voting(all_secrets)
                if corrections_count > 0:
                    # 计算投票后的相似度
                    if self.expected_secret:
                        _, voted_similarity = self.core.compare_secrets(
                            self.expected_secret, voted_secret, self.similarity_threshold
                        )
                    else:
                        voted_similarity = 1.0
                    
                    # 计算投票前的平均相似度
                    avg_similarity_before = np.mean([sim for _, sim in extracted_results])
                    
                    # 如果投票纠错后满足阈值
                    if voted_similarity >= self.similarity_threshold:
                        result['success'] = True
                        result['secret'] = voted_secret
                        result['similarity'] = voted_similarity
                        result['method'] = 'sliding_window'
                        result['vote_ratio'] = 1.0
                        result['window_size'] = window_size
                        result['error_correction_used'] = True
                        result['error_correction_type'] = 'voting'
                        result['corrected_bits'] = corrections_count
                        result['similarity_before_correction'] = avg_similarity_before
                        result['similarity_after_correction'] = voted_similarity
                        return result
            
            # 过滤出满足阈值的结果
            valid_results = [(s, sim) for s, sim in extracted_results 
                            if sim >= self.similarity_threshold]
            
            if valid_results:
                # 步骤c: 投票决定最终水印
                voted_secret, max_similarity, vote_ratio = self._vote_secret(valid_results)
                
                # 步骤d: 当前大小窗口提取到水印则成功返回
                result['success'] = True
                result['secret'] = voted_secret
                result['similarity'] = max_similarity
                result['method'] = 'sliding_window'
                result['vote_ratio'] = vote_ratio
                result['window_size'] = window_size
                return result
        
        # 步骤e: 所有常规方法都失败，尝试预处理后再提取
        if use_multiscale:
            preprocessing_methods = [
                ('jpeg_enhancement', self._preprocess_jpeg_enhancement, 'JPEG伪去压缩'),
                ('nlmeans_denoise', self._preprocess_nlmeans_denoise, '非局部均值去噪'),
                ('combined', self._preprocess_combined, 'JPEG增强+去噪')
            ]
            
            for preprocess_name, preprocess_func, preprocess_desc in preprocessing_methods:
                try:
                    # 应用预处理
                    preprocessed_image = preprocess_func(image)
                    
                    # 对预处理后的图像使用多尺度窗口提取
                    for window_size in self.WINDOW_SIZES:
                        extracted_results = self._extract_window(preprocessed_image, window_size)
                        
                        if not extracted_results:
                            continue
                        
                        # 尝试投票纠错（如果启用）
                        all_secrets = [s for s, _ in extracted_results]
                        if enable_error_correction and len(all_secrets) > 1:
                            voted_secret, corrections_count, voting_details = self.core.error_correction_voting(all_secrets)
                            if corrections_count > 0:
                                # 计算投票后的相似度
                                if self.expected_secret:
                                    _, voted_similarity = self.core.compare_secrets(
                                        self.expected_secret, voted_secret, self.similarity_threshold
                                    )
                                else:
                                    voted_similarity = 1.0
                                
                                # 计算投票前的平均相似度
                                avg_similarity_before = np.mean([sim for _, sim in extracted_results])
                                
                                # 如果投票纠错后满足阈值
                                if voted_similarity >= self.similarity_threshold:
                                    result['success'] = True
                                    result['secret'] = voted_secret
                                    result['similarity'] = voted_similarity
                                    result['method'] = f'sliding_window_preprocessed_{preprocess_name}'
                                    result['vote_ratio'] = 1.0
                                    result['window_size'] = window_size
                                    result['preprocessing'] = preprocess_desc
                                    result['error_correction_used'] = True
                                    result['error_correction_type'] = 'voting'
                                    result['corrected_bits'] = corrections_count
                                    result['similarity_before_correction'] = avg_similarity_before
                                    result['similarity_after_correction'] = voted_similarity
                                    return result
                        
                        # 过滤出满足阈值的结果
                        valid_results = [(s, sim) for s, sim in extracted_results 
                                        if sim >= self.similarity_threshold]
                        
                        if valid_results:
                            # 投票决定最终水印
                            voted_secret, max_similarity, vote_ratio = self._vote_secret(valid_results)
                            
                            # 预处理后成功提取
                            result['success'] = True
                            result['secret'] = voted_secret
                            result['similarity'] = max_similarity
                            result['method'] = f'sliding_window_preprocessed_{preprocess_name}'
                            result['vote_ratio'] = vote_ratio
                            result['window_size'] = window_size
                            result['preprocessing'] = preprocess_desc
                            return result
                        
                except Exception as e:
                    # 预处理失败，跳过该方法
                    continue
        
        # 对于小图像，最后尝试填充方式（如果之前的填充尝试失败）
        w, h = image.size
        if (w < self.target_size or h < self.target_size) and not result['success']:
            pad_secret, pad_similarity = self._pad_and_extract(image)
            
            # 尝试汉明纠错（如果启用且有期望密钥）
            if enable_error_correction and self.expected_secret and pad_similarity < self.similarity_threshold:
                corrected_secret, corrected_bits, did_correction = self.core.error_correction_hamming(
                    pad_secret, self.expected_secret, max_hamming_corrections
                )
                if did_correction:
                    pad_secret = corrected_secret
                    _, pad_similarity = self.core.compare_secrets(
                        self.expected_secret, pad_secret, self.similarity_threshold
                    )
                    if pad_similarity >= self.similarity_threshold:
                        result['success'] = True
                        result['secret'] = pad_secret
                        result['similarity'] = pad_similarity
                        result['method'] = 'pad_final'
                        result['error_correction_used'] = True
                        result['error_correction_type'] = 'hamming'
                        result['corrected_bits'] = corrected_bits
                        result['similarity_before_correction'] = original_similarity
                        result['similarity_after_correction'] = pad_similarity
                        return result
            
            if pad_similarity >= self.similarity_threshold:
                result['success'] = True
                result['secret'] = pad_secret
                result['similarity'] = pad_similarity
                result['method'] = 'pad_final'
                return result
        
        # 所有方法都失败，返回resize方法的原始结果（可能是纠错后的）
        result['secret'] = secret
        result['similarity'] = similarity
        result['method'] = 'resize'
        return result
    
    def extract_file(self, 
                     input_path: str,
                     use_multiscale: bool = True,
                     enable_error_correction: bool = True,
                     max_hamming_corrections: int = 3) -> Dict:
        """
        从单个图像文件提取水印
        
        Args:
            input_path: 输入图像路径
            use_multiscale: 是否使用多尺度滑动窗口
            enable_error_correction: 是否启用纠错功能
            max_hamming_corrections: 汉明纠错最大允许纠正的比特数
            
        Returns:
            Dict: 提取结果
        """
        try:
            result = self.extract(input_path, use_multiscale, enable_error_correction, max_hamming_corrections)
            result['file'] = input_path
            return result
        except Exception as e:
            return {
                'success': False,
                'secret': None,
                'similarity': 0.0,
                'method': None,
                'vote_ratio': 0.0,
                'window_size': None,
                'file': input_path,
                'error': str(e),
                'error_correction_used': False,
                'error_correction_type': None,
                'corrected_bits': 0,
                'similarity_before_correction': 0.0,
                'similarity_after_correction': 0.0
            }
    
    def extract_folder(self, 
                       input_folder: str,
                       recursive: bool = True,
                       use_multiscale: bool = True,
                       enable_error_correction: bool = True,
                       max_hamming_corrections: int = 3,
                       show_progress: bool = True) -> Dict:
        """
        从文件夹中的所有图像提取水印（支持嵌套文件夹）
        
        Args:
            input_folder: 输入文件夹路径
            recursive: 是否递归处理子文件夹
            use_multiscale: 是否使用多尺度滑动窗口
            enable_error_correction: 是否启用纠错功能
            max_hamming_corrections: 汉明纠错最大允许纠正的比特数
            show_progress: 是否显示进度条
            
        Returns:
            Dict: 处理结果统计
        """
        input_folder = Path(input_folder)
        
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
            'details': []
        }
        
        iterator = tqdm(image_files, desc="提取水印") if show_progress else image_files
        
        for img_path in iterator:
            file_result = self.extract_file(str(img_path), use_multiscale, enable_error_correction, max_hamming_corrections)
            results['details'].append(file_result)
            
            if file_result['success']:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def extract_batch(self, 
                      input_path: str,
                      recursive: bool = True,
                      use_multiscale: bool = True,
                      show_progress: bool = True) -> Dict:
        """
        统一的批量处理接口，自动判断输入是文件还是文件夹
        
        Args:
            input_path: 输入路径（文件或文件夹）
            recursive: 是否递归处理子文件夹（仅对文件夹有效）
            use_multiscale: 是否使用多尺度滑动窗口
            show_progress: 是否显示进度条
            
        Returns:
            Dict: 处理结果
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            # 单文件处理
            result = self.extract_file(str(input_path), use_multiscale)
            return {
                'total': 1,
                'success': 1 if result['success'] else 0,
                'failed': 0 if result['success'] else 1,
                'details': [result]
            }
        elif input_path.is_dir():
            # 文件夹处理
            return self.extract_folder(
                str(input_path), recursive, use_multiscale, show_progress
            )
        else:
            raise ValueError(f"无效的输入路径: {input_path}")
    
    def verify(self, 
               image: Union[str, Image.Image, np.ndarray],
               expected_secret: str,
               use_multiscale: bool = True) -> Dict:
        """
        验证图像中是否包含指定的水印
        
        Args:
            image: 图像路径、PIL图像或numpy数组
            expected_secret: 期望的水印密钥
            use_multiscale: 是否使用多尺度滑动窗口
            
        Returns:
            Dict: 验证结果
        """
        # 临时设置期望密钥
        original_expected = self.expected_secret
        self.expected_secret = expected_secret
        
        result = self.extract(image, use_multiscale)
        
        # 恢复原来的期望密钥
        self.expected_secret = original_expected
        
        return result
