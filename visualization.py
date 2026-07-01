import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

def save_masks(original_tensor, mask_tensor,filename):
    mask_tensor = mask_tensor.argmax(1)[0]
    # image1 = Image.fromarray(mask_array[0])
    original_tensor = original_tensor[0]
    # 2. 扩展掩码的维度，使其与原图的通道数匹配
    mask_tensor = mask_tensor.expand_as(original_tensor)

    # 3. 融合原图与掩码
    #原图反标准化
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    original_tensor = original_tensor * std[:, None, None] + mean[:, None, None]
    alpha = 0.5  # 融合强度，可以调整
    fused_tensor = original_tensor * (1 - mask_tensor) + mask_tensor * torch.tensor([1, 0, 0]).view(3, 1, 1) * alpha  # 这里将掩码融合为红色

    # 4. 转换为NumPy数组以便保存
    fused_tensor = torch.clamp(fused_tensor, 0, 1)  # 确保值在 [0, 1] 范围
    fused_image = (fused_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # 转换为 (H, W, C) 形状并乘以255
    fused_image = Image.fromarray(fused_image)
# 保存为PNG格式
    fused_image.save(f"./visualization/{filename}.PNG")
    



def save_images(img_tensor,filename):
    # 定义标准化的均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # 反标准化
    img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]

    # 确保值在[0, 1]范围内
    img_tensor = torch.clamp(img_tensor, 0, 1)

    # 转换为NumPy数组
    img_np = img_tensor.permute(1, 2, 0).numpy()  # 将通道从 (C, H, W) 转换为 (H, W, C)

    # 转换为PIL图像
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))  # 将像素值转换为[0, 255]范围

    # 保存图像
    img_pil.save(filename)


def load_vocab(filename):
    vocab = {}
    with open(filename, 'r') as f:
        for index, line in enumerate(f):
            word = line.strip()
            vocab[index] = word  
    return vocab

def save_sentences(encoded, vocab, filename, batch_i):
    encoded = encoded.numpy()[0]
    sentence = ' '.join(vocab[token.item()] for token in encoded if token.item() != 0)
    with open(filename, 'a') as f:
        f.write(f'image{batch_i}: ' + sentence + '\n')


def view_attention_map(features, original_image, filename):
    # 可视化注意力图并与原图融合
    #fig, axes = plt.subplots(1, len(attention_maps), figsize=(16, 4))
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = original_image * std[:, None, None] + mean[:, None, None]
    #print(img.shape)
    img = img.squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    H, W, C = img.shape
    #x_c1, x_c2, x_c3, x_c4 = features
    #print(x_c1.shape, x_c2.shape, x_c3.shape, x_c4.shape)
    attention_maps = [torch.sum(f.cpu(), dim=1) for f in features]
    attn_maps = np.ones((H,W))
    for i, attn_map in enumerate(attention_maps):
        # 将注意力图从Tensor转换为NumPy，并调整大小
        #print(attn_map.shape)
        attn_map = attn_map.squeeze(0).cpu().detach().numpy()
        attn_map = np.interp(attn_map, (attn_map.min(), attn_map.max()), (0, 1)) # 归一化到[0,1]
        attn_map = cv2.resize(attn_map, (H, W))
        attn_maps = attn_maps * attn_map
        attn_maps = np.interp(attn_maps, (attn_maps.min(), attn_maps.max()), (0, 1))
    attn_maps = (attn_maps * 255).astype(np.uint8)
        # 将原图转换为NumPy格式
        #original_img = original_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        #original_img = np.interp(original_img, (original_img.min(), original_img.max()), (0, 1))  # 归一化到[0,1]
        
        # 生成热图并叠加
    heatmap = cv2.applyColorMap(attn_maps, cv2.COLORMAP_JET) # 使用Jet colormap生成热图
        #print(heatmap.shape)
        #fused_image = 0.7 * img + 0.3 * heatmap  # 融合原图和热图（0.7为权重）
    fused_image = cv2.addWeighted(img, 1, heatmap, 0.5, 0)
    cv2.imwrite(f"./att_image/{filename}.PNG", fused_image)
        #转换为NumPy数组以便保存
        #fused_tensor = torch.clamp(fused_tensor, 0, 1)  # 确保值在 [0, 1] 范围
        #fused_image = (fused_image * 255).astype(np.uint8)  # 转换为 (H, W, C) 形状并乘以255
        #fused_image = Image.fromarray(fused_image)
        #fused_image.save(f"./att_image/{filename}_scale{i}.PNG")
