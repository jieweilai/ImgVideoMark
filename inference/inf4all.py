# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist
import torch.nn as nn
import os
import io
import pickle
import numpy as np
from Stage1_Model import MessageExtractor as Decoder, MessageProcessor as Encoder
from model import Embedder, Extractor
from torchvision.utils import save_image
from einops import rearrange
from PIL import Image, ImageOps
from torchvision import transforms

class _DDPUnpickler(pickle.Unpickler):
    """自定义 Unpickler，绕过 DDP 反序列化问题"""
    def __init__(self, file, map_location):
        super().__init__(file)
        self.map_location = map_location
        
    def find_class(self, module, name):
        # 当遇到 DistributedDataParallel 时，返回一个代理类
        if name == 'DistributedDataParallel':
            return _DDPProxy
        return super().find_class(module, name)
    
    def persistent_load(self, pid):
        # 处理 torch tensor 的加载
        return torch.serialization._legacy_load(pid, self.map_location)

class _DDPProxy:
    """DDP 代理类，用于在反序列化时提取内部模型"""
    def __init__(self, *args, **kwargs):
        pass
    
    def __setstate__(self, state):
        # 只保存 module 的 state_dict
        if 'module' in state:
            self._module = state['module']
        else:
            self._module = None
        self._state = state
    
    def state_dict(self):
        if self._module is not None:
            return self._module.state_dict()
        return {}

def load_state_dict_from_ddp(model, model_path, device='cuda:0'):
    """加载 DDP 训练保存的模型，去掉 module. 前缀"""
    import zipfile
    
    device_obj = torch.device(device)
    
    # 尝试用自定义方式加载，绕过 DDP 问题
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            # 查找 data.pkl 文件
            pkl_file = None
            for name in zf.namelist():
                if name.endswith('data.pkl'):
                    pkl_file = name
                    break
            
            if pkl_file:
                # 直接从 zip 中读取 state_dict 相关的 tensor
                # 使用 torch 的内置方法但带有自定义处理
                pass
    except:
        pass
    
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
    model = model.to(device)
    model.eval()
    return model

def enc():
    model_path = 'models/encoder.pth'
    encoder = Encoder()
    encoder = load_state_dict_from_ddp(encoder, model_path)
    
    # generate random bit string with length 30 for testing #
    bit_length = 30
    bit_array = np.random.randint(0, 2, size=bit_length)
    
    secret_input = torch.tensor(bit_array, dtype=torch.float32).cuda()
    
    secret_input = secret_input.unsqueeze(0)
    
    
    with torch.no_grad():
        secret_pattern = encoder(secret_input)
    
    return secret_pattern, secret_input


def dec(pattern):
    model_path = 'models/decoder.pth'

    decoder = Decoder()
    new_state = {}
    decoder_dict = torch.load(model_path)
    
    for key, value in decoder_dict.items():
        new_state[key.replace('module.', '')] = value
        
    decoder.load_state_dict(new_state)
    decoder = decoder.cuda()
    decoder.eval()

    with torch.no_grad():
        secret_output = decoder(pattern)
    
    return torch.round(secret_output)



def embed():
    model_path = 'saved_models/embedder.pth'
    embedder = Embedder()
    embedder = load_state_dict_from_ddp(embedder, model_path)

    # Read I_ori in #
    to_tensor = transforms.ToTensor()
    img_ori_path = 'I_ori.png'


    I_ori = Image.open(img_ori_path).convert('RGB')
    I_ori = ImageOps.fit(I_ori, (200, 200))
    I_ori = to_tensor(I_ori).cuda()
    I_ori = I_ori.unsqueeze(0)

    secret_pattern, secret_input = enc()

    I_w = embedder((secret_pattern, I_ori)) + I_ori
    I_w = torch.clamp(I_w, 0, 1)
    save_image(I_w, 'I_w.png')
    print(secret_input)

    return secret_input


def partial_theft():
    I_w = np.array(Image.open('I_w.png'))
    I_bg = np.array(Image.open('I_bg.png'))
    I_mask = np.array(Image.open('I_mask.png'))
    
    if len(I_mask.shape) == 3:
        I_mask = I_mask[:, :, 0] if I_mask.shape[2] >= 1 else I_mask.mean(axis=2)
    
    mask_input = I_mask.astype(np.float32) / 255.0
    
    if len(I_w.shape) == 3:
        mask_input = np.expand_dims(mask_input, axis=2)
    
    I_compo = mask_input * I_w + (1 - mask_input) * I_bg

    I_compo = I_compo.astype(np.uint8)
    Image.fromarray(I_compo).save('I_compo.png')
    
    return I_compo



def extract():
    model_path = 'saved_models/extractor.pth'
    extractor = Extractor()
    extractor = load_state_dict_from_ddp(extractor, model_path)

    # Read I_compo (or I_w) in #
    to_tensor = transforms.ToTensor()
    img_path = 'I_compo.png'


    I_todecode = Image.open(img_path).convert('RGB')
    I_todecode = to_tensor(I_todecode).cuda()
    I_todecode = I_todecode.unsqueeze(0)

    extracted_pattern = extractor(I_todecode)
    decoded_secret = dec(extracted_pattern)
    print(decoded_secret)

    return decoded_secret



if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '19889'
    # dist.init_process_group(backend='nccl', rank=0, world_size = 1)
    # dist.init_process_group(backend='gloo', rank=0, world_size = 1)
    
    print(torch.__version__)
    
    secret_input = embed()
    partial_theft()
    secret_output = extract()