"""
Copy-Move攻击性能测试脚本

测试不同裁剪比例和缩放比例下的Copy-Move攻击对水印的影响
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from func.watermark_embedder import WatermarkEmbedder
from func.watermark_extractor import WatermarkExtractor

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class ImageProcessor:
    """图像处理器，提供Copy-Move功能"""
    
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
        else:
            paste_x = 0
            paste_y = 0
        
        # 粘贴
        result.paste(scaled, (paste_x, paste_y))
        
        return result


class CopyMovePerformanceTester:
    """Copy-Move攻击性能测试器"""
    
    def __init__(self, 
                 embedder: WatermarkEmbedder,
                 extractor: WatermarkExtractor,
                 base_output_dir: str = '../img/results_copy_move',
                 enable_error_correction: bool = True):
        """
        初始化测试器
        
        Args:
            embedder: 水印嵌入器
            extractor: 水印提取器
            base_output_dir: 输出目录
            enable_error_correction: 是否启用纠错功能
        """
        self.embedder = embedder
        self.extractor = extractor
        self.processor = ImageProcessor()
        self.base_output_dir = Path(base_output_dir)
        self.enable_error_correction = enable_error_correction
        
        # 创建输出目录
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'test_name': 'Copy-Move Performance Test',
            'timestamp': datetime.now().isoformat(),
            'watermark_secret': self.embedder.get_secret_string(),
            'error_correction_enabled': enable_error_correction,
            'test_config': {},
            'results': []
        }
    
    def test_copy_move_grid(self, 
                           source_images: List[str],
                           target_images: List[str],
                           crop_ratios: List[float] = None,
                           scale_factors: List[float] = None) -> Dict:
        """
        测试Copy-Move攻击在不同参数组合下的性能
        
        Args:
            source_images: 源图像路径列表
            target_images: 目标图像路径列表
            crop_ratios: 裁剪比例列表，默认[0.1, 0.2, ..., 0.9]
            scale_factors: 缩放因子列表，默认[0.1, 0.2, ..., 2.0]
            
        Returns:
            Dict: 测试结果
        """
        # 默认参数
        if crop_ratios is None:
            crop_ratios = [round(x * 0.1, 1) for x in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
        if scale_factors is None:
            scale_factors = [round(x * 0.1, 1) for x in range(1, 21)]  # [0.1, 0.2, ..., 2.0]
        
        print("\n" + "=" * 60)
        print("Copy-Move攻击性能测试")
        print("=" * 60)
        print(f"源图像数量: {len(source_images)}")
        print(f"目标图像数量: {len(target_images)}")
        print(f"裁剪比例: {crop_ratios}")
        print(f"缩放因子: {scale_factors}")
        print(f"总测试数: {len(source_images) * len(target_images) * len(crop_ratios) * len(scale_factors)}")
        print(f"纠错功能: {'启用' if self.enable_error_correction else '禁用'}")
        print("=" * 60)
        
        # 保存配置
        self.results['test_config'] = {
            'source_images': [str(p) for p in source_images],
            'target_images': [str(p) for p in target_images],
            'crop_ratios': crop_ratios,
            'scale_factors': scale_factors,
            'num_source_images': len(source_images),
            'num_target_images': len(target_images),
            'num_crop_ratios': len(crop_ratios),
            'num_scale_factors': len(scale_factors),
            'total_tests': len(source_images) * len(target_images) * len(crop_ratios) * len(scale_factors)
        }
        
        # 创建保存目录
        save_dir = self.base_output_dir / 'images'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        test_count = 0
        success_count = 0
        
        # 遍历所有组合
        for source_path in source_images:
            source_name = Path(source_path).stem
            source_img = Image.open(source_path).convert('RGB')
            
            # 嵌入水印到源图像
            watermarked_source = self.embedder.embed(source_img)
            
            for target_path in target_images:
                target_name = Path(target_path).stem
                target_img = Image.open(target_path).convert('RGB')
                
                print(f"\n源图像: {source_name} → 目标图像: {target_name}")
                
                for crop_ratio in crop_ratios:
                    for scale_factor in scale_factors:
                        test_count += 1
                        
                        print(f"  [{test_count}/{self.results['test_config']['total_tests']}] "
                              f"裁剪={crop_ratio}, 缩放={scale_factor}... ", end='')
                        
                        try:
                            # 执行Copy-Move攻击
                            cm_result = self.processor.copy_move(
                                watermarked_source, 
                                target_img, 
                                crop_ratio=crop_ratio,
                                scale=scale_factor,
                                seed=42
                            )
                            
                            # 保存结果图像（可选，为节省空间可以注释掉）
                            # filename = f'{source_name}_to_{target_name}_crop{crop_ratio}_scale{scale_factor}.png'
                            # cm_result.save(save_dir / filename)
                            
                            # 提取水印
                            extract_result = self.extractor.extract(
                                cm_result, 
                                enable_error_correction=self.enable_error_correction
                            )
                            
                            # 记录结果
                            test_result = {
                                'source_image': source_name,
                                'target_image': target_name,
                                'crop_ratio': crop_ratio,
                                'scale_factor': scale_factor,
                                'success': extract_result['success'],
                                'similarity': extract_result.get('similarity', 0),
                                'method': extract_result.get('method', 'unknown'),
                                'error_correction_used': extract_result.get('error_correction_used', False),
                                'error_correction_type': extract_result.get('error_correction_type'),
                                'corrected_bits': extract_result.get('corrected_bits', 0),
                                'similarity_before_correction': extract_result.get('similarity_before_correction'),
                                'similarity_after_correction': extract_result.get('similarity_after_correction'),
                                'preprocessing': extract_result.get('preprocessing')
                            }
                            
                            self.results['results'].append(test_result)
                            
                            if extract_result['success']:
                                success_count += 1
                                print(f"✓ (相似度: {extract_result['similarity']:.2%})")
                            else:
                                print(f"✗ (相似度: {extract_result['similarity']:.2%})")
                        
                        except Exception as e:
                            print(f"✗ 错误: {str(e)}")
                            test_result = {
                                'source_image': source_name,
                                'target_image': target_name,
                                'crop_ratio': crop_ratio,
                                'scale_factor': scale_factor,
                                'success': False,
                                'similarity': 0,
                                'error': str(e)
                            }
                            self.results['results'].append(test_result)
        
        # 计算汇总统计
        self.results['summary'] = {
            'total_tests': test_count,
            'success_count': success_count,
            'failed_count': test_count - success_count,
            'success_rate': success_count / test_count if test_count > 0 else 0
        }
        
        # 统计纠错使用情况
        if self.enable_error_correction:
            ec_used = sum(1 for r in self.results['results'] if r.get('error_correction_used', False))
            self.results['summary']['error_correction_used_count'] = ec_used
            self.results['summary']['error_correction_used_rate'] = ec_used / test_count if test_count > 0 else 0
        
        # 统计预处理使用情况
        preprocess_used = sum(1 for r in self.results['results'] if r.get('preprocessing') is not None)
        self.results['summary']['preprocessing_used_count'] = preprocess_used
        self.results['summary']['preprocessing_used_rate'] = preprocess_used / test_count if test_count > 0 else 0
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        print(f"总测试数: {test_count}")
        print(f"成功数: {success_count}")
        print(f"失败数: {test_count - success_count}")
        print(f"成功率: {success_count / test_count * 100:.2f}%")
        
        if self.enable_error_correction:
            print(f"纠错使用次数: {ec_used} ({ec_used / test_count * 100:.1f}%)")
        if preprocess_used > 0:
            print(f"预处理使用次数: {preprocess_used} ({preprocess_used / test_count * 100:.1f}%)")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_path: str = None):
        """
        保存测试结果到JSON文件
        
        Args:
            output_path: 输出文件路径，默认为base_output_dir/results.json
        """
        if output_path is None:
            output_path = self.base_output_dir / 'results.json'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_path}")
        return output_path
    
    def visualize_heatmap(self, save_path: str = None):
        """
        可视化Copy-Move攻击性能热力图
        
        Args:
            save_path: 保存路径，默认为base_output_dir/heatmap.png
        """
        if not self.results['results']:
            print("没有测试结果可以可视化！")
            return
        
        crop_ratios = sorted(set(r['crop_ratio'] for r in self.results['results']))
        scale_factors = sorted(set(r['scale_factor'] for r in self.results['results']))
        
        # 创建热力图矩阵（成功率）
        heatmap_data = np.zeros((len(crop_ratios), len(scale_factors)))
        
        for i, crop_ratio in enumerate(crop_ratios):
            for j, scale_factor in enumerate(scale_factors):
                # 获取该参数组合的所有测试结果
                tests = [r for r in self.results['results'] 
                        if r['crop_ratio'] == crop_ratio and r['scale_factor'] == scale_factor]
                if tests:
                    success_count = sum(1 for t in tests if t['success'])
                    heatmap_data[i, j] = success_count / len(tests) * 100
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 10))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # 设置坐标轴
        ax.set_xticks(np.arange(len(scale_factors)))
        ax.set_yticks(np.arange(len(crop_ratios)))
        ax.set_xticklabels([f'{s:.1f}' for s in scale_factors])
        ax.set_yticklabels([f'{c:.1f}' for c in crop_ratios])
        
        # 设置标签
        ax.set_xlabel('缩放因子', fontsize=14, fontweight='bold')
        ax.set_ylabel('裁剪比例', fontsize=14, fontweight='bold')
        
        # 添加标题
        title = f'Copy-Move攻击性能热力图\n'
        title += f'总成功率: {self.results["summary"]["success_rate"]*100:.1f}% '
        title += f'({self.results["summary"]["success_count"]}/{self.results["summary"]["total_tests"]})'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 在每个单元格中显示数值
        for i in range(len(crop_ratios)):
            for j in range(len(scale_factors)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.0f}%',
                              ha="center", va="center", color="black", fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('成功率 (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = self.base_output_dir / 'heatmap.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Copy-Move攻击性能测试')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='测试图像目录')
    parser.add_argument('--output', '-o', type=str, default='../img/results_copy_move',
                       help='输出目录 (默认: ../img/results_copy_move)')
    parser.add_argument('--secret', '-s', type=str, default=None,
                       help='30位二进制水印密钥 (默认: 随机生成)')
    parser.add_argument('--json', '-j', type=str, default='results.json',
                       help='JSON结果文件名 (默认: results.json)')
    parser.add_argument('--crop-start', type=float, default=0.1,
                       help='裁剪比例起始值 (默认: 0.1)')
    parser.add_argument('--crop-end', type=float, default=0.9,
                       help='裁剪比例结束值 (默认: 0.9)')
    parser.add_argument('--crop-step', type=float, default=0.1,
                       help='裁剪比例间隔 (默认: 0.1)')
    parser.add_argument('--scale-start', type=float, default=0.1,
                       help='缩放因子起始值 (默认: 0.1)')
    parser.add_argument('--scale-end', type=float, default=2.0,
                       help='缩放因子结束值 (默认: 2.0)')
    parser.add_argument('--scale-step', type=float, default=0.1,
                       help='缩放因子间隔 (默认: 0.1)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='批处理大小 (默认: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='运行设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 查找图像
    img_dir_path = Path(args.input)
    if not img_dir_path.exists():
        print(f"错误: 目录不存在 {args.input}")
        return
    
    image_files = sorted(list(img_dir_path.glob('*.png')) + 
                        list(img_dir_path.glob('*.jpg')) + 
                        list(img_dir_path.glob('*.jpeg')))
    
    if not image_files:
        print(f"错误: 目录中没有找到图像文件")
        return
    
    print(f"\n找到 {len(image_files)} 张图像")
    
    # 将图像分为源图像和目标图像（前一半为源，后一半为目标）
    mid = max(1, len(image_files) // 2)
    source_images = [str(f) for f in image_files[:mid]]
    target_images = [str(f) for f in image_files[mid:mid*2]]
    
    # 确保至少有一对
    if not target_images:
        target_images = source_images[:]
    
    # 生成参数列表
    crop_ratios = [round(x, 2) for x in np.arange(args.crop_start, args.crop_end + args.crop_step/2, args.crop_step)]
    scale_factors = [round(x, 2) for x in np.arange(args.scale_start, args.scale_end + args.scale_step/2, args.scale_step)]
    
    json_path = str(Path(args.output) / args.json)
    
    # 显示配置
    print("\n" + "=" * 60)
    print("Copy-Move攻击性能测试")
    print("=" * 60)
    print(f"源图像: {len(source_images)}张")
    print(f"目标图像: {len(target_images)}张")
    print(f"裁剪比例: {len(crop_ratios)}个 [{crop_ratios[0]}~{crop_ratios[-1]}]")
    print(f"缩放因子: {len(scale_factors)}个 [{scale_factors[0]}~{scale_factors[-1]}]")
    print(f"总测试数: {len(source_images) * len(target_images) * len(crop_ratios) * len(scale_factors)}")
    print(f"批处理大小: {args.batch_size}")
    print(f"设备: {args.device}")
    print(f"输出: {json_path}")
    print(f"纠错: 启用")
    print("=" * 60)
    
    # 初始化组件
    print("\n初始化水印系统...")
    embedder = WatermarkEmbedder(
        watermark_secret=args.secret,
        device=args.device
    )
    extractor = WatermarkExtractor(
        expected_secret=embedder.get_secret_string(),
        device=args.device
    )
    
    print(f"水印密钥: {embedder.get_secret_string()}")
    
    # 设置batch_size（虽然当前实现中单个测试处理，但为将来优化预留）
    embedder.core.batch_size = args.batch_size
    extractor.core.batch_size = args.batch_size
    
    # 创建测试器
    tester = CopyMovePerformanceTester(
        embedder=embedder,
        extractor=extractor,
        base_output_dir=args.output,
        enable_error_correction=True
    )
    
    # 运行测试
    results = tester.test_copy_move_grid(
        source_images=source_images,
        target_images=target_images,
        crop_ratios=crop_ratios,
        scale_factors=scale_factors
    )
    
    # 保存结果
    tester.save_results(json_path)
    
    # 可视化
    heatmap_path = Path(args.output) / 'heatmap.png'
    tester.visualize_heatmap(str(heatmap_path))


if __name__ == '__main__':
    main()
