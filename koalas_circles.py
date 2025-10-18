"""
Koalas Circles 节点 - 将图像转换为圆形像素点表示

该节点将图像分解为圆形像素点，类似于Koalas to the Max效果。
支持标准模式和随机模式，可以设置细分级别范围以及圆形半径系数。
随机模式下会智能保留图像主体区域的细节。
"""

import numpy as np
import torch
from PIL import Image, ImageDraw
# import comfy.utils  # 未使用，注释避免不必要依赖
import math
import random
from scipy import ndimage

# ---------------------------
# 使用中英文混合：内部仍采用英文参数，界面显示名中文
# ---------------------------

# ============================ 核心实现 ============================

class KoalasCirclesCore:
    """将图像转换为圆形像素点表示的ComfyUI节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """使用中文字段名以在 UI 中显示中文"""
        return {
            "required": {
                "图像": ("IMAGE",),
                "模式": (["standard", "random"], {"default": "random"}),
                "最小级别": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "最大级别": ("INT", {"default": 7, "min": 1, "max": 8, "step": 1}),
                "随机性": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "细节阈值": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01}),
                "主体聚焦": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "半径系数": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "背景颜色": ("STRING", {"default": "#000000"}),
                "透明背景": ("BOOLEAN", {"default": False}),
                "保持原尺寸": ("BOOLEAN", {"default": False}),
                "输出宽度": ("INT", {"default": 1024, "min": -1, "max": 4096, "step": 64}),
                "输出高度": ("INT", {"default": 1024, "min": -1, "max": 4096, "step": 64}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"  # 仅供兼容，核心类不会被直接注册

    def _preprocess_image_tensor(self, tensor):
        """预处理图像张量，处理各种异常情况"""
        # 打印原始形状以便调试
        original_shape = tensor.shape
        print(f"原始图像张量形状: {original_shape}")
        
        # 检查是否是标准的批次、高度、宽度、通道格式
        if len(tensor.shape) == 4 and tensor.shape[0] == 1 and tensor.shape[1] > 1 and tensor.shape[2] > 1 and tensor.shape[3] in [1, 3, 4]:
            # 标准格式，直接返回
            return tensor
        
        # 特殊处理(1, 1, 1, 1)形状 - 单个值
        if len(tensor.shape) == 4 and tensor.shape[1] == 1 and tensor.shape[2] == 1 and tensor.shape[3] == 1:
            print("检测到单个值形状(1, 1, 1, 1)，创建均匀颜色图像")
            # 获取单个值
            value = tensor[0, 0, 0, 0].item()
            # 归一化到0-1
            if value > 1.0:
                value = value / 255.0
            # 创建16x16的均匀颜色图像
            new_tensor = torch.ones((1, 16, 16, 3), device=tensor.device) * value
            return new_tensor
        
        # 特殊处理(1, 1, 1, 450)形状
        if len(tensor.shape) == 4 and tensor.shape[1] == 1 and tensor.shape[2] == 1 and tensor.shape[3] == 450:
            print("检测到特殊形状(1, 1, 1, 450)，应用专门处理")
            
            # 创建一个21x21的新张量(因为21²=441，接近450)
            width, height = 21, 21
            new_tensor = torch.zeros((1, height, width, 3), device=tensor.device)
            
            try:
                # 从一维数据创建二维图像
                values = tensor[0, 0, 0, :441]  # 取前441个值
                reshaped = values.reshape(height, width)
                
                # 归一化值到0-1范围
                if reshaped.max() > 1.0:
                    reshaped = reshaped / 255.0
                
                # 复制到三个通道
                for c in range(3):
                    new_tensor[0, :, :, c] = reshaped
                
                print("成功将(1, 1, 1, 450)数据重塑为21x21图像")
                return new_tensor
            except Exception as e:
                print(f"重塑(1, 1, 1, 450)数据失败: {e}")
        
        # 处理其他(1, 1, 1, N)形状
        if len(tensor.shape) == 4 and tensor.shape[1] == 1 and tensor.shape[2] == 1:
            total_values = tensor.shape[3]
            print(f"处理一维数据形状(1, 1, 1, {total_values})")
            
            # 尝试转换为接近正方形的图像
            side = int(math.sqrt(total_values))
            width = side
            height = side
            
            # 如果不是完美平方数，调整尺寸
            if width * height < total_values:
                width = side + 1
            if width * height < total_values:
                height = side + 1
                
            # 创建新张量
            new_tensor = torch.zeros((1, height, width, 3), device=tensor.device)
            
            try:
                # 尝试将一维数据重塑为二维图像
                if total_values >= width * height:
                    values = tensor[0, 0, 0, :width*height]
                    reshaped = values.reshape(height, width)
                    
                    # 归一化值到0-1范围
                    if reshaped.max() > 1.0:
                        reshaped = reshaped / 255.0
                    
                    # 复制到三个通道
                    for c in range(3):
                        new_tensor[0, :, :, c] = reshaped
                else:
                    # 数据点不足，填充部分数据
                    values = tensor[0, 0, 0, :]
                    flat_values = torch.zeros(width * height, device=tensor.device)
                    flat_values[:total_values] = values
                    reshaped = flat_values.reshape(height, width)
                    
                    # 归一化值到0-1范围
                    if reshaped.max() > 1.0:
                        reshaped = reshaped / 255.0
                    
                    # 复制到三个通道
                    for c in range(3):
                        new_tensor[0, :, :, c] = reshaped
                
                print(f"成功将一维数据重塑为{width}x{height}图像")
                return new_tensor
            except Exception as e:
                print(f"重塑一维数据失败: {e}")
        
        # 处理其他异常形状
        try:
            print(f"尝试处理异常形状: {tensor.shape}")
            
            # 如果是单通道，转为三通道
            if len(tensor.shape) == 4 and tensor.shape[3] == 1:
                tensor = tensor.repeat(1, 1, 1, 3)
                
            # 如果通道数过多，只取前三个
            if len(tensor.shape) == 4 and tensor.shape[3] > 3:
                tensor = tensor[:, :, :, :3]
                
            # 如果没有批次维度，添加批次维度
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
                
            # 如果形状仍然异常，创建默认图像
            if len(tensor.shape) != 4 or tensor.shape[1] <= 1 or tensor.shape[2] <= 1:
                raise ValueError(f"无法处理形状 {tensor.shape}")
                
            return tensor
        except Exception as e:
            print(f"处理异常形状失败: {e}")
        
        # 其他情况，创建一个默认的灰色图像
        print(f"无法处理形状 {tensor.shape}，创建默认图像")
        default_tensor = torch.ones((1, 64, 64, 3), device=tensor.device) * 0.5
        return default_tensor

    def run(self, 图像, 模式, 最小级别, 最大级别, 随机性, 细节阈值, 主体聚焦, 半径系数, 背景颜色, 透明背景, 保持原尺寸, 输出宽度, 输出高度, 随机种子):
        # -------- 参数映射 --------
        image_tensor = 图像
        mode = 模式
        min_level = max(2, 最小级别) if mode == "random" else 1  # 随机模式要求最小>=2
        max_level = 最大级别
        randomness = 随机性
        detail_threshold = 细节阈值
        subject_focus = 主体聚焦
        radius_rate = 半径系数
        bg_color = 背景颜色
        keep_original = 保持原尺寸
        out_w = 输出宽度
        out_h = 输出高度
        seed = 随机种子 if mode == "random" else 0
        transparent_bg = 透明背景  # 使用传入的透明背景参数
        
        # 处理背景颜色
        if bg_color == "transparent":
            transparent_bg = True
        elif not bg_color.startswith('#'):
            bg_color = f"#{bg_color}"
        
        if seed == -1:
            seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        self.rng = random.Random(seed)

        # --- 解析输入图像: [B, H, W, C] (最终修正版) ---
        try:
            # ComfyUI 的标准图像格式是 [Batch, Height, Width, Channel]
            b, h, w, c = image_tensor.shape
            print(f"输入图像张量形状: [B={b}, H={h}, W={w}, C={c}]")
            orig_h, orig_w = h, w

            # 从批次中取第一张图，已经是 [H, W, C] 的 numpy 格式
            img_np = image_tensor[0].cpu().numpy()

            # 转换为 PIL Image 所需的 uint8 格式
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)

            img_pil = Image.fromarray(img_np)
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            print(f"成功从 [H, W, C] 创建PIL图像: {img_pil.size}, mode={img_pil.mode}")

        except Exception as e:
            print(f"解析图像张量失败: {e}")
            import traceback
            traceback.print_exc()
            img_pil = Image.new('RGB', (512, 512), (128, 128, 128))
            orig_h, orig_w = 512, 512
            print("使用灰色备用图像")

        # --- 保持原始比例并确定处理尺寸 ---
        orig_width, orig_height = img_pil.size
        print(f"原始图像尺寸: {orig_width}x{orig_height}")
        
        # 计算目标尺寸，保持原始比例
        if keep_original:
            target_width, target_height = orig_width, orig_height
        else:
            target_width, target_height = out_w, out_h
        
        print(f"目标输出尺寸: {target_width}x{target_height}")
        
        # 计算处理尺寸，找到最接近2的幂次方的尺寸
        # 使用较大维度来确定基本尺寸，保持原始比例
        max_dim = max(orig_width, orig_height)
        base_power = math.floor(math.log2(max_dim))
        base_size = 2 ** base_power
        
        # 计算保持原始比例的新尺寸
        if orig_width >= orig_height:
            process_width = base_size
            process_height = int(base_size * (orig_height / orig_width))
        else:
            process_height = base_size
            process_width = int(base_size * (orig_width / orig_height))
        
        # 确保高度和宽度至少为2^3=8
        process_width = max(process_width, 8)
        process_height = max(process_height, 8)
        
        print(f"处理尺寸: {process_width}x{process_height}")
        
        # --- 调整图像到处理尺寸 ---
        img_pil = img_pil.resize((process_width, process_height), Image.LANCZOS)
        
        # --- 超采样抗锯齿 ---
        upscale = 2
        draw_width = process_width * upscale
        draw_height = process_height * upscale
        img_pil = img_pil.resize((draw_width, draw_height), Image.LANCZOS)
        img_array = np.array(img_pil)
        # 始终在 RGBA 画布上绘制，便于透明处理
        result = Image.new("RGBA", (draw_width, draw_height), (0, 0, 0, 0) if transparent_bg else bg_color)
        draw = ImageDraw.Draw(result, "RGBA")

        # --- 核心处理 ---
        if mode == "standard":
            self._draw_uniform_grid(img_array, draw, max_level, radius_rate)
        else:
            # 使用新的四叉树随机算法 v2
            self.subject_map = None
            if subject_focus > 0:
                self.subject_map = self._detect_subject(img_array, subject_focus)
            root_size = max(draw_width, draw_height)
            root = self._make_node(img_array, 0, 0, root_size, 0)
            self._process_node_random_v2(root, draw, min_level, max_level, randomness, detail_threshold, subject_focus, radius_rate, img_array)

        # --- 抗锯齿：缩回到处理尺寸 ---
        result = result.resize((process_width, process_height), Image.LANCZOS)

        # --- 调整至最终输出尺寸 ---
        if keep_original:
            final_w, final_h = orig_w, orig_h
        else:
            final_w, final_h = out_w, out_h
            
        # 保持纵横比，避免变形
        if (final_w, final_h) != result.size:
            # 计算缩放比例，保持纵横比
            scale_w = final_w / process_width
            scale_h = final_h / process_height
            scale = min(scale_w, scale_h)
            
            # 计算缩放后的尺寸
            scaled_w = int(process_width * scale)
            scaled_h = int(process_height * scale)
            
            # 先按比例缩放
            result = result.resize((scaled_w, scaled_h), Image.LANCZOS)
            
            # 如果需要，创建画布并居中放置图像
            if scaled_w != final_w or scaled_h != final_h:
                canvas = Image.new("RGB", (final_w, final_h), bg_color)
                paste_x = (final_w - scaled_w) // 2
                paste_y = (final_h - scaled_h) // 2
                canvas.paste(result, (paste_x, paste_y))
                result = canvas

        # --- 转回 ComfyUI 张量 [B, H, W, C] ---
        # ComfyUI IMAGE 张量目前仅支持 3 通道 RGB
        # 若用户选择透明背景，则在输出前将透明通道合成到指定背景色，避免 ComfyUI 显示异常
        if transparent_bg:
            try:
                comp = Image.new("RGBA", result.size, (0, 0, 0, 0))
                comp.alpha_composite(result)
                result_rgb = comp.convert("RGB")
            except Exception as e:
                print(f"alpha_composite 失败: {e}, 直接转换 RGB")
                result_rgb = result.convert("RGB")
        else:
            result_rgb = result.convert("RGB")

        np_out = np.array(result_rgb).astype(np.float32) / 255.0
        tensor_out = torch.from_numpy(np_out).unsqueeze(0)
        print(f"输出张量形状: {tensor_out.shape}")
        return (tensor_out,)
    
    def _detect_subject(self, img_array, subject_focus):
        """检测图像中的主体区域，返回一个权重图"""
        # 转换为灰度图
        gray = np.mean(img_array, axis=2)
        
        # 计算边缘强度
        dx = ndimage.sobel(gray, axis=0)
        dy = ndimage.sobel(gray, axis=1)
        edge_strength = np.sqrt(dx**2 + dy**2)
        
        # 计算局部对比度
        local_std = ndimage.generic_filter(gray, np.std, size=15)
        
        # 计算亮度变化
        brightness = gray / 255.0
        brightness_weight = 1.0 - np.abs(brightness - 0.5) * 2  # 中等亮度区域权重高
        
        # 归一化边缘强度和局部对比度，避免除以零
        edge_norm = edge_strength / (edge_strength.max() + 1e-8)
        local_norm = local_std / (local_std.max() + 1e-8)

        subject_weight = (
            edge_norm * 0.5 +
            local_norm * 0.3 +
            brightness_weight * 0.2
        )
        
        # 平滑处理
        subject_weight = ndimage.gaussian_filter(subject_weight, sigma=5)
        
        # 归一化
        subject_weight = (subject_weight - subject_weight.min()) / (subject_weight.max() - subject_weight.min())
        
        # 应用非线性变换，增强主体区域的权重
        subject_weight = subject_weight ** 0.5
        
        return subject_weight
    
    def _get_subject_importance(self, x, y, width, height):
        """获取区域的主体重要性分数，安全处理边界情况"""
        if self.subject_map is None or width <= 0 or height <= 0:
            return 0.5
            
        try:
            region = self.subject_map[y:y+height, x:x+width]
            if region.size == 0:
                return 0.5
            return np.mean(region)
        except Exception:
            return 0.5
    
    def _sample_color(self, img_array, x, y, w, h):
        """采样区域内的平均颜色"""
        # 确保区域有效
        if x >= img_array.shape[1] or y >= img_array.shape[0] or w <= 0 or h <= 0:
            return (128, 128, 128)  # 返回灰色
            
        # 确保区域在图像范围内
        x_end = min(x + w, img_array.shape[1])
        y_end = min(y + h, img_array.shape[0])
        
        # 如果区域为空，返回灰色
        if x >= x_end or y >= y_end:
            return (128, 128, 128)
            
        try:
            region = img_array[y:y_end, x:x_end]
            if region.size == 0:  # 如果区域为空
                return (128, 128, 128)
                
            r = np.mean(region[:, :, 0])
            g = np.mean(region[:, :, 1])
            b = np.mean(region[:, :, 2])
            
            # 检查是否有NaN值
            if np.isnan(r) or np.isnan(g) or np.isnan(b):
                return (128, 128, 128)
                
            return (int(r), int(g), int(b))
        except Exception as e:
            print(f"采样颜色时出错: {e}")
            return (128, 128, 128)  # 返回灰色
    
    def _make_node(self, img_array, x, y, s, level):
        """创建一个节点，包含位置、大小、层级和颜色信息"""
        return {
            'x': x, 
            'y': y, 
            'size': s, 
            'level': level,
            'color': self._sample_color(img_array, x, y, s, s),
            'children': None
        }
    
    def _subdivide(self, node, img_array, max_level):
        """将节点细分为四个子节点，确保始终能创建子节点"""
        # 如果已有子节点或达到最大级别，则不再细分
        if node['children'] or node['level'] >= max_level:
            return
        
        # 获取图像尺寸
        img_height, img_width = img_array.shape[:2]
        
        # 获取节点信息
        node_x, node_y = node['x'], node['y']
        node_size = node['size']
        
        # 计算子节点尺寸 - 始终是父节点的一半
        half_size = max(2, node_size // 2)  # 确保至少为2
        next_level = node['level'] + 1
        
        # 创建四个子节点 - 不过早判断边界
        children = []
        
        # 左上
        children.append(self._make_node(img_array, node_x, node_y, half_size, next_level))
        
        # 右上
        children.append(self._make_node(img_array, node_x + half_size, node_y, half_size, next_level))
        
        # 左下
        children.append(self._make_node(img_array, node_x, node_y + half_size, half_size, next_level))
        
        # 右下
        children.append(self._make_node(img_array, node_x + half_size, node_y + half_size, half_size, next_level))
        
        # 设置子节点
        node['children'] = children
    
    def _calculate_variation(self, img_array, x, y, width, height):
        """计算区域内的颜色变化程度 - RGB版本，安全处理边界情况"""
        if width <= 1 or height <= 1:
            return 0
            
        try:
            region = img_array[y:y+height, x:x+width]
            if region.size == 0:
                return 0
                
            std_r = np.std(region[:, :, 0]) / 255.0
            std_g = np.std(region[:, :, 1]) / 255.0
            std_b = np.std(region[:, :, 2]) / 255.0
            
            return (std_r + std_g + std_b) / 3
        except Exception:
            return 0
    
    def _process_node_random_v2(self, node, draw, min_level, max_level, randomness, detail_threshold, subject_focus, radius_rate, img_array):
        """改进版四叉树随机细分：
        - variation < 0.5*threshold : 直接停 → 大圆
        - variation > 1.5*threshold : 必细分
        - 中间区域：按随机性概率决定
        主体区域更不容易合并（更细分）。"""
        x, y, size, level = node['x'], node['y'], node['size'], node['level']
        # 终止条件1：已到最大深度或size过小
        if level >= max_level or size <= 2:
            self._draw_circle(node, draw, radius_rate)
            return

        h, w = img_array.shape[:2]
        # 有效区域
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + size), min(h, y + size)
        if x2 <= x1 or y2 <= y1:
            return

        variation = self._calculate_variation(img_array, x1, y1, x2 - x1, y2 - y1)
        subject_importance = 0.5
        if hasattr(self, 'subject_map') and self.subject_map is not None and subject_focus > 0:
            subject_importance = self._get_subject_importance(x1, y1, x2 - x1, y2 - y1) * subject_focus
            subject_importance = max(0.0, min(1.0, subject_importance))

        # 必须细分的条件
        must_split = False
        if level < min_level:
            must_split = True
        elif variation > detail_threshold * 1.5:
            must_split = True

        # 可合并的条件
        can_merge = variation < detail_threshold * 0.5

        if not must_split:
            if can_merge:
                should_split = False
            else:
                # 中间区域 -> 随机决定，主体越重要越倾向细分
                stop_prob = (detail_threshold * 1.5 - variation) / (detail_threshold)
                stop_prob = max(0.0, min(1.0, stop_prob))
                # 调整：随机性越大 → 越倾向细分；主体越重要 → 越倾向细分
                split_prob = (1 - randomness) * 0.6 + subject_importance * 0.2 + 0.2
                should_split = self.rng.random() < split_prob
        else:
            should_split = True

        if should_split:
            self._subdivide(node, img_array, max_level)
            if node['children']:
                for child in node['children']:
                    self._process_node_random_v2(child, draw, min_level, max_level, randomness, detail_threshold, subject_focus, radius_rate, img_array)
            else:
                # 叶节点：仅使用固定半径系数
                self._draw_circle(node, draw, radius_rate)
        else:
            # 叶节点：仅使用固定半径系数
            self._draw_circle(node, draw, radius_rate)

    def _draw_circle(self, node, draw, radius_rate):
        """在画布上绘制圆形"""
        # 计算圆心和半径
        cx = node['x'] + node['size'] / 2
        cy = node['y'] + node['size'] / 2
        r = node['size'] / 2 * radius_rate
        
        # 获取画布尺寸
        img_width, img_height = draw.im.size
        
        # 只有当圆形完全在画布外时才跳过
        if (cx + r < 0 or cx - r > img_width or
            cy + r < 0 or cy - r > img_height):
            return
        
        # 计算圆形边界，但不严格限制在画布内
        # 允许圆形稍微超出画布边缘，避免图像不完整
        left = cx - r
        top = cy - r
        right = cx + r
        bottom = cy + r
        
        # 绘制圆形
        try:
            draw.ellipse((left, top, right, bottom), fill=node['color'])
        except Exception:
            # 如果出错，尝试限制在画布范围内再绘制
            left = max(0, left)
            top = max(0, top)
            right = min(img_width, right)
            bottom = min(img_height, bottom)
            
            # 确保有效的圆形区域
            if right > left and bottom > top:
                draw.ellipse((left, top, right, bottom), fill=node['color'])
    
    def _draw_uniform_grid(self, img_array, draw, level, radius_rate):
        """在 draw 上绘制统一大小的圆，根据 level (2^level 网格)，支持非正方形图像"""
        h, w = img_array.shape[:2]
        
        # 计算水平和垂直方向的网格数，根据图像尺寸可能不同
        grid_h = 2 ** level
        grid_w = 2 ** level
        
        # 计算单元格尺寸
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        # 确保单元格尺寸至少为1
        cell_h = max(1, cell_h)
        cell_w = max(1, cell_w)
        
        # 计算半径，使用较小的单元格尺寸确保圆不会重叠
        r = min(cell_h, cell_w) * radius_rate / 2.0
        
        # 绘制圆形网格
        for gy in range(grid_h):
            y0 = gy * cell_h
            if y0 >= h:
                continue
                
            for gx in range(grid_w):
                x0 = gx * cell_w
                if x0 >= w:
                    continue
                    
                # 计算单元格区域，确保不超出图像边界
                y_end = min(y0 + cell_h, h)
                x_end = min(x0 + cell_w, w)
                
                # 采样该格子平均颜色
                region = img_array[y0:y_end, x0:x_end]
                if region.size == 0:
                    continue
                    
                color = (
                    int(np.mean(region[:, :, 0])),
                    int(np.mean(region[:, :, 1])),
                    int(np.mean(region[:, :, 2])),
                )
                
                # 计算圆心
                cx = x0 + (x_end - x0) / 2.0
                cy = y0 + (y_end - y0) / 2.0
                
                # 绘制圆形
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

    def _draw_random_grid(self, img_array, draw, level, randomness, detail_threshold, subject_focus, radius_rate):
        """在 draw 上绘制随机大小的圆，圆心位于统一网格中。
        level 决定网格分辨率：网格数 = 2 ** level
        圆半径由颜色方差、主体重要性与随机性共同决定。"""
        h, w = img_array.shape[:2]
        grid_h = 2 ** level
        grid_w = 2 ** level
        cell_h = h // grid_h
        cell_w = w // grid_w
        cell_h = max(2, cell_h)
        cell_w = max(2, cell_w)

        # 若需要主体重要性图
        subj_map = None
        if subject_focus > 0:
            subj_map = self._detect_subject(img_array, subject_focus)

        for gy in range(grid_h):
            y0 = gy * cell_h
            if y0 >= h:
                continue
            y_end = min(y0 + cell_h, h)
            for gx in range(grid_w):
                x0 = gx * cell_w
                if x0 >= w:
                    continue
                x_end = min(x0 + cell_w, w)

                # 区域像素
                region = img_array[y0:y_end, x0:x_end]
                if region.size == 0:
                    continue

                # 平均颜色
                color = (
                    int(np.mean(region[:, :, 0])),
                    int(np.mean(region[:, :, 1])),
                    int(np.mean(region[:, :, 2])),
                )

                # 颜色方差(0~1)
                var = self._calculate_variation(img_array, x0, y0, x_end - x0, y_end - y0)

                # 主体重要性(0~1)
                subj = 0.5
                if subj_map is not None:
                    subj_region = subj_map[y0:y_end, x0:x_end]
                    if subj_region.size > 0:
                        subj = float(np.mean(subj_region))
                        subj *= subject_focus
                        subj = max(0.0, min(1.0, subj))

                # 基础半径
                base_r = min(cell_h, cell_w) * radius_rate / 2.0

                # 方差因子: 变化越大 -> 圆越小
                var_factor = 1.0 - var  # 高方差 -> 0.0
                var_factor = 0.3 + 0.7 * var_factor

                # 主体因子: 主体越重要 -> 圆越小
                subj_factor = 1.0 - 0.7 * subj  # subj=1 -> 0.3, subj=0 ->1.0

                # 随机因子
                rand_span = 0.4 * randomness  # 最大 ±40% 调整
                rand_factor = 1.0 + rand_span * (self.rng.random() * 2 - 1)

                r = base_r * var_factor * subj_factor * rand_factor
                r = max(1.0, r)  # 不要太小

                # 圆心
                cx = x0 + (x_end - x0) / 2.0
                cy = y0 + (y_end - y0) / 2.0

                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

# ============================ 包装节点 - 标准 ============================

class KoalasCirclesStandard:
    """标准模式节点（固定 mode=standard，隐藏随机性相关参数）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "级别": ("INT", {"default": 7, "min": 1, "max": 8, "step": 1}),
                "细节阈值": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01}),
                "半径系数": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "背景颜色": ("STRING", {"default": "#000000", "widget": "color"}),
                "透明背景": ("BOOLEAN", {"default": False}),
                "保持原尺寸": ("BOOLEAN", {"default": False}),
                "输出宽度": ("INT", {"default": 1024, "min": -1, "max": 4096, "step": 64}),
                "输出高度": ("INT", {"default": 1024, "min": -1, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, 图像, 级别, 细节阈值, 半径系数, 背景颜色, 透明背景, 保持原尺寸, 输出宽度, 输出高度):
        # 处理背景颜色
        if not isinstance(背景颜色, str):
            背景颜色 = "#000000"
        else:
            背景颜色 = 背景颜色.strip().lower()
            if 背景颜色 in ["false", "true", "", "none"]:
                背景颜色 = "#000000"
            elif 背景颜色 == "transparent":
                背景颜色 = "transparent"
            elif not 背景颜色.startswith('#'):
                # 若形如 "ff00ff" 或 "f0f" 等，自动补 '#'
                背景颜色 = f"#{背景颜色}"
            
        # 校验输出尺寸，防止小于最小要求导致验证失败
        if not isinstance(输出宽度, int) or 输出宽度 < 64:
            输出宽度 = 图像.shape[2] if 保持原尺寸 else 1024
        if not isinstance(输出高度, int) or 输出高度 < 64:
            输出高度 = 图像.shape[1] if 保持原尺寸 else 1024
        core = KoalasCirclesCore()
        level_int = int(级别)
        # 传递一个固定的种子（例如0）以确保确定性
        return core.run(图像, "standard", 1, level_int, 1.0, 细节阈值, 0.0, 半径系数, 背景颜色, 透明背景, 保持原尺寸, 输出宽度, 输出高度, 0)

# ============================ 包装节点 - 随机 ============================

class KoalasCirclesRandom:
    """随机模式节点（固定 mode=random，显示随机性与主体聚焦）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "最小级别": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "最大级别": ("INT", {"default": 7, "min": 1, "max": 8, "step": 1}),
                "随机性": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "细节阈值": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01}),
                "主体聚焦": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "半径系数": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "背景颜色": ("STRING", {"default": "#000000", "widget": "color"}),
                "透明背景": ("BOOLEAN", {"default": False}),
                "保持原尺寸": ("BOOLEAN", {"default": False}),
                "输出宽度": ("INT", {"default": 1024, "min": -1, "max": 4096, "step": 64}),
                "输出高度": ("INT", {"default": 1024, "min": -1, "max": 4096, "step": 64}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, 图像, 最小级别, 最大级别, 随机性, 细节阈值, 主体聚焦, 半径系数, 背景颜色, 透明背景, 保持原尺寸, 输出宽度, 输出高度, 随机种子):
        # 处理背景颜色
        if not isinstance(背景颜色, str):
            背景颜色 = "#000000"
        else:
            背景颜色 = 背景颜色.strip().lower()
            if 背景颜色 in ["false", "true", "", "none"]:
                背景颜色 = "#000000"
            elif 背景颜色 == "transparent":
                背景颜色 = "transparent"
            elif not 背景颜色.startswith('#'):
                # 若形如 "ff00ff" 或 "f0f" 等，自动补 '#'
                背景颜色 = f"#{背景颜色}"

            
        # 校验输出尺寸，防止小于最小要求导致验证失败
        if not isinstance(输出宽度, int) or 输出宽度 < 64:
            输出宽度 = 图像.shape[2] if 保持原尺寸 else 1024
        if not isinstance(输出高度, int) or 输出高度 < 64:
            输出高度 = 图像.shape[1] if 保持原尺寸 else 1024
        core = KoalasCirclesCore()
        return core.run(图像, "random", 最小级别, 最大级别, 随机性, 细节阈值, 主体聚焦, 半径系数, 背景颜色, 透明背景, 保持原尺寸, 输出宽度, 输出高度, 随机种子)

# ============================ 颜色选择器节点 ============================

# ============================ 颜色选择器节点已移除 ============================
# 颜色选择功能已直接集成到圆形节点中

# ============================ 注册 ============================

NODE_CLASS_MAPPINGS = {
    "KoalasCirclesStandard": KoalasCirclesStandard,
    "KoalasCirclesRandom": KoalasCirclesRandom,
    # 添加与云端配置匹配的名称
    "koalas_circles": KoalasCirclesRandom,  # 默认使用随机模式
    "koalas_circles_standard": KoalasCirclesStandard,
    "koalas_circles_random": KoalasCirclesRandom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KoalasCirclesStandard": "Koalas 圆形（标准）",
    "KoalasCirclesRandom": "Koalas 圆形（随机）",
    # 添加与云端配置匹配的显示名称
    "koalas_circles": "Koalas 圆形",
    "koalas_circles_standard": "Koalas 圆形（标准）",
    "koalas_circles_random": "Koalas 圆形（随机）",
}