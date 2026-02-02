# -*- coding: utf-8 -*-
"""
水印核心模块 - 提供基础的水印嵌入和提取功能
与 inf4all.py 保持一致的模型加载方式
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, Union, List, Dict,Counter

# 添加inference模块路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inference'))

from Stage1_Model import MessageExtractor as Decoder, MessageProcessor as Encoder
from model import Embedder, Extractor


class WatermarkCore:
    """水印核心处理类，封装基础的嵌入和提取操作"""
    
    # 模型支持的图像大小
    SUPPORTED_SIZE = 200
    # 水印比特长度
    BIT_LENGTH = 30
    
    def __init__(self, 
                 encoder_path: Optional[str] = None,
                 decoder_path: Optional[str] = None,
                 embedder_path: Optional[str] = None,
                 extractor_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化水印核心模块
        
        Args:
            encoder_path: 编码器模型路径
            decoder_path: 解码器模型路径
            embedder_path: 嵌入器模型路径
            extractor_path: 提取器模型路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = device
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inference')
        
        # 设置默认模型路径
        self.encoder_path = encoder_path or os.path.join(base_path, 'models', 'encoder.pth')
        self.decoder_path = decoder_path or os.path.join(base_path, 'models', 'decoder.pth')
        self.embedder_path = embedder_path or os.path.join(base_path, 'saved_models', 'embedder.pth')
        self.extractor_path = extractor_path or os.path.join(base_path, 'saved_models', 'extractor.pth')
        
        # 延迟加载模型
        self._encoder = None
        self._decoder = None
        self._embedder = None
        self._extractor = None
        
        # 图像转换
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def _load_state_dict_from_ddp(self, model, model_path: str):
        """
        加载 DDP 训练保存的模型，与 inf4all.py 保持一致
        使用 monkey-patch 方式绕过 DDP 反序列化问题
        """
        device_obj = torch.device(self.device)
        
        # 使用 monkey-patch 方式加载
        original_setstate = None
        if hasattr(nn.parallel.DistributedDataParallel, '__setstate__'):
            original_setstate = nn.parallel.DistributedDataParallel.__setstate__
        
        def patched_setstate(self, state):
            # 最小化的 setstate，只恢复必要的属性
            self.__dict__.update(state)
        
        try:
            nn.parallel.DistributedDataParallel.__setstate__ = patched_setstate
            loaded = torch.load(model_path, map_location=device_obj)
        finally:
            if original_setstate:
                nn.parallel.DistributedDataParallel.__setstate__ = original_setstate
        
        # 如果是完整模型对象（DDP保存的），尝试获取 state_dict
        if hasattr(loaded, 'module'):
            state_dict = loaded.module.state_dict()
        elif hasattr(loaded, 'state_dict'):
            state_dict = loaded.state_dict()
        else:
            state_dict = loaded
        
        # 去掉 module. 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        model.eval()
        return model
    
    @property
    def encoder(self):
        """延迟加载编码器 - 使用 MessageProcessor"""
        if self._encoder is None:
            encoder = Encoder()
            self._encoder = self._load_state_dict_from_ddp(encoder, self.encoder_path)
        return self._encoder
    
    @property
    def decoder(self):
        """延迟加载解码器 - 使用 MessageExtractor"""
        if self._decoder is None:
            decoder = Decoder()
            decoder_dict = torch.load(self.decoder_path, map_location=self.device)
            new_state = {}
            for key, value in decoder_dict.items():
                new_state[key.replace('module.', '')] = value
            decoder.load_state_dict(new_state)
            decoder = decoder.to(self.device)
            decoder.eval()
            self._decoder = decoder
        return self._decoder
    
    @property
    def embedder(self):
        """延迟加载嵌入器 - 使用 Embedder"""
        if self._embedder is None:
            embedder = Embedder()
            self._embedder = self._load_state_dict_from_ddp(embedder, self.embedder_path)
        return self._embedder
    
    @property
    def extractor(self):
        """延迟加载提取器 - 使用 Extractor"""
        if self._extractor is None:
            extractor = Extractor()
            self._extractor = self._load_state_dict_from_ddp(extractor, self.extractor_path)
        return self._extractor
    
    def generate_secret(self, bit_string: Optional[str] = None) -> torch.Tensor:
        """
        生成或解析水印密钥
        
        Args:
            bit_string: 30位的二进制字符串，如 "101010..."，如果为None则随机生成
            
        Returns:
            torch.Tensor: 密钥张量
        """
        if bit_string is None:
            bit_array = np.random.randint(0, 2, size=self.BIT_LENGTH)
        else:
            if len(bit_string) != self.BIT_LENGTH:
                raise ValueError(f"水印密钥必须是{self.BIT_LENGTH}位二进制字符串")
            bit_array = np.array([int(b) for b in bit_string])
        
        secret_input = torch.tensor(bit_array, dtype=torch.float32).to(self.device)
        secret_input = secret_input.unsqueeze(0)
        return secret_input
    
    def encode_secret(self, secret_input: torch.Tensor) -> torch.Tensor:
        """
        将密钥编码为水印图案
        
        Args:
            secret_input: 密钥张量
            
        Returns:
            torch.Tensor: 水印图案
        """
        with torch.no_grad():
            secret_pattern = self.encoder(secret_input)
        return secret_pattern
    
    def decode_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        从水印图案解码密钥
        
        Args:
            pattern: 水印图案 [B, 1, 200, 200]
            
        Returns:
            torch.Tensor: 解码后的密钥（四舍五入后的二进制）[B, 30]
        """
        with torch.no_grad():
            secret_output = self.decoder(pattern)
        return torch.round(secret_output)
    
    def embed_watermark_single(self, 
                               image: torch.Tensor, 
                               secret_pattern: torch.Tensor) -> torch.Tensor:
        """
        在单个200x200图像上嵌入水印
        
        Args:
            image: 图像张量 [1, 3, 200, 200]
            secret_pattern: 水印图案
            
        Returns:
            torch.Tensor: 嵌入水印后的图像
        """
        with torch.no_grad():
            residual = self.embedder((secret_pattern, image))
            watermarked = residual + image
            watermarked = torch.clamp(watermarked, 0, 1)
        return watermarked
    
    def embed_watermark_batch(self, 
                              images: torch.Tensor, 
                              secret_pattern: torch.Tensor) -> torch.Tensor:
        """
        批量嵌入水印
        
        Args:
            images: 图像张量 [B, 3, 200, 200]
            secret_pattern: 水印图案 [1, 1, 200, 200]
            
        Returns:
            torch.Tensor: 嵌入水印后的图像 [B, 3, 200, 200]
        """
        batch_size = images.shape[0]
        # 扩展 secret_pattern 到 batch_size
        secret_pattern_batch = secret_pattern.expand(batch_size, -1, -1, -1)
        
        with torch.no_grad():
            residual = self.embedder((secret_pattern_batch, images))
            watermarked = residual + images
            watermarked = torch.clamp(watermarked, 0, 1)
        return watermarked
    
    def extract_pattern_single(self, image: torch.Tensor) -> torch.Tensor:
        """
        从单个200x200图像中提取水印图案
        
        Args:
            image: 图像张量 [1, 3, 200, 200]
            
        Returns:
            torch.Tensor: 提取的水印图案
        """
        with torch.no_grad():
            pattern = self.extractor(image)
        return pattern
    
    def extract_pattern_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        批量提取水印图案
        
        Args:
            images: 图像张量 [B, 3, 200, 200]
            
        Returns:
            torch.Tensor: 提取的水印图案 [B, 1, 200, 200]
        """
        with torch.no_grad():
            patterns = self.extractor(images)
        return patterns
    
    def extract_secret_single(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        从单个200x200图像中提取水印密钥
        
        Args:
            image: 图像张量 [1, 3, 200, 200]
            
        Returns:
            Tuple[torch.Tensor, str]: (密钥张量, 密钥字符串)
        """
        pattern = self.extract_pattern_single(image)
        secret = self.decode_pattern(pattern)
        secret_str = ''.join([str(int(b)) for b in secret[0].cpu().numpy()])
        return secret, secret_str
    
    def extract_secret_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        批量提取水印密钥
        
        Args:
            images: 图像张量 [B, 3, 200, 200]
            
        Returns:
            Tuple[torch.Tensor, List[str]]: (密钥张量 [B, 30], 密钥字符串列表)
        """
        patterns = self.extract_pattern_batch(images)
        secrets = self.decode_pattern(patterns)
        secret_strs = []
        for i in range(secrets.shape[0]):
            secret_str = ''.join([str(int(b)) for b in secrets[i].cpu().numpy()])
            secret_strs.append(secret_str)
        return secrets, secret_strs
    
    def images_to_batch_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        """
        将多个PIL图像转换为批量张量
        
        Args:
            images: PIL图像列表
            
        Returns:
            torch.Tensor: 批量图像张量 [B, 3, H, W]
        """
        tensors = []
        for img in images:
            tensor = self.to_tensor(img)
            tensors.append(tensor)
        batch = torch.stack(tensors, dim=0).to(self.device)
        return batch
    
    def batch_tensor_to_images(self, tensor: torch.Tensor) -> List[Image.Image]:
        """
        将批量张量转换为PIL图像列表
        
        Args:
            tensor: 批量图像张量 [B, 3, H, W]
            
        Returns:
            List[Image.Image]: PIL图像列表
        """
        tensor = tensor.cpu().clamp(0, 1)
        images = []
        for i in range(tensor.shape[0]):
            img = self.to_pil(tensor[i])
            images.append(img)
        # 显式删除CPU张量
        del tensor
        return images
    
    def image_to_tensor(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        将图像转换为张量
        
        Args:
            image: 图像路径、PIL图像或numpy数组
            
        Returns:
            torch.Tensor: 图像张量
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        tensor = self.to_tensor(image).to(self.device)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor
    
    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        将张量转换为PIL图像
        
        Args:
            tensor: 图像张量
            
        Returns:
            Image.Image: PIL图像
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.cpu().clamp(0, 1)
        return self.to_pil(tensor)
    
    def secret_to_string(self, secret: torch.Tensor) -> str:
        """将密钥张量转换为字符串"""
        if secret.dim() == 2:
            secret = secret[0]
        return ''.join([str(int(b)) for b in secret.cpu().numpy()])
    
    def compare_secrets(self, secret1: str, secret2: str, threshold: float = 0.9) -> Tuple[bool, float]:
        """
        比较两个密钥的相似度
        
        Args:
            secret1: 第一个密钥字符串
            secret2: 第二个密钥字符串
            threshold: 相似度阈值
            
        Returns:
            Tuple[bool, float]: (是否匹配, 相似度比例)
        """
        if len(secret1) != len(secret2):
            return False, 0.0
        
        matches = sum(a == b for a, b in zip(secret1, secret2))
        similarity = matches / len(secret1)
        is_match = similarity >= threshold
        return is_match, similarity
    
    def hamming_distance(self, secret1: str, secret2: str) -> int:
        """
        计算两个密钥的汉明距离
        
        Args:
            secret1: 第一个密钥字符串
            secret2: 第二个密钥字符串
            
        Returns:
            int: 汉明距离（不同比特数）
        """
        if len(secret1) != len(secret2):
            return len(secret1)
        return sum(a != b for a, b in zip(secret1, secret2))
    
    def error_correction_hamming(self, extracted_secret: str, expected_secret: str, 
                                 max_corrections: int = 3) -> Tuple[str, int, bool]:
        """
        基于汉明距离的纠错（单比特翻转纠错）
        如果提取的密钥与期望密钥的汉明距离小于等于max_corrections，则纠正为期望值
        
        Args:
            extracted_secret: 提取的密钥字符串
            expected_secret: 期望的密钥字符串
            max_corrections: 最大允许纠正的比特数
            
        Returns:
            Tuple[str, int, bool]: (纠错后的密钥, 纠正的比特数, 是否进行了纠错)
        """
        if len(extracted_secret) != len(expected_secret):
            return extracted_secret, 0, False
        
        distance = self.hamming_distance(extracted_secret, expected_secret)
        
        # 如果汉明距离在允许范围内，纠正为期望值
        if 0 < distance <= max_corrections:
            return expected_secret, distance, True
        
        # 否则不纠错
        return extracted_secret, distance, False
    
    def error_correction_voting(self, secrets_list: List[str]) -> Tuple[str, int, Dict]:
        """
        基于投票的纠错（多次提取投票）
        对于每个比特位置，选择出现次数最多的值
        
        Args:
            secrets_list: 多个提取的密钥字符串列表
            
        Returns:
            Tuple[str, int, Dict]: (纠错后的密钥, 使用了纠错的比特数, 投票详情)
        """
        if not secrets_list:
            return '', 0, {}
        
        if len(secrets_list) == 1:
            return secrets_list[0], 0, {}
        
        bit_length = len(secrets_list[0])
        corrected_secret = []
        corrections_count = 0
        voting_details = {}
        
        for bit_pos in range(bit_length):
            # 收集该位置的所有比特值
            bits = [secret[bit_pos] for secret in secrets_list if len(secret) > bit_pos]
            
            # 投票
            counter = Counter(bits)
            most_common_bit, count = counter.most_common(1)[0]
            
            corrected_secret.append(most_common_bit)
            
            # 如果不是所有提取都一致，说明进行了纠错
            if len(counter) > 1:
                corrections_count += 1
                voting_details[bit_pos] = {
                    'votes': dict(counter),
                    'chosen': most_common_bit,
                    'confidence': count / len(bits)
                }
        
        corrected_secret_str = ''.join(corrected_secret)
        return corrected_secret_str, corrections_count, voting_details
