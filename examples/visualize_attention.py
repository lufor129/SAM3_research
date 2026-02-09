# SAM3 Visual Prompt Attention Visualization
# 此腳本展示如何在 cross-image few-shot detection 中視覺化 attention

"""
目標: 了解 Support Image 的 BBox 區域在 Query Image 上關注了哪些位置

關鍵位置:
1. Transformer Encoder: cross-attention between visual prompt and image features
2. Transformer Decoder: cross-attention between queries and encoded features

我們需要 hook 進 attention 層來獲取 attention weights
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============================================================================
# Attention Hook 類別
# ============================================================================

class AttentionHook:
    """
    Hook 用於捕捉 MultiheadAttention 的 attention weights
    """
    def __init__(self):
        self.attention_weights = []
        self.hooks = []
        
    def clear(self):
        self.attention_weights = []
        
    def hook_fn(self, module, inputs, outputs):
        """
        nn.MultiheadAttention 的 forward 輸出:
        - outputs[0]: attention output
        - outputs[1]: attention weights (if need_weights=True)
        
        但 SAM3 預設 need_weights=False，所以我們需要用另一種方法
        """
        # 儲存 module 資訊以便後續分析
        self.attention_weights.append({
            'module': module,
            'inputs': inputs,
            'outputs': outputs
        })
    
    def register_hooks(self, model):
        """註冊 hooks 到所有 attention 層"""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
                print(f"Registered hook on: {name}")
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# 手動計算 Attention Weights 的方法
# ============================================================================

@torch.inference_mode()
def compute_cross_attention_manually(query, key, value, num_heads=8):
    """
    手動計算 cross-attention weights
    
    Args:
        query: (seq_len, batch, embed_dim)
        key: (seq_len, batch, embed_dim)
        value: (seq_len, batch, embed_dim)
        num_heads: number of attention heads
        
    Returns:
        attention_weights: (batch, num_heads, query_len, key_len)
    """
    seq_len_q, batch, embed_dim = query.shape
    seq_len_k = key.shape[0]
    head_dim = embed_dim // num_heads
    
    # Reshape for multi-head attention
    q = query.transpose(0, 1)  # (batch, seq_len, embed_dim)
    k = key.transpose(0, 1)
    
    # Reshape to (batch, num_heads, seq_len, head_dim)
    q = q.view(batch, seq_len_q, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len_k, num_heads, head_dim).transpose(1, 2)
    
    # Compute attention scores
    scale = head_dim ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    return attn_weights


# ============================================================================
# 修改版的 grounding_with_visual_prompt (帶 attention 輸出)
# ============================================================================

@torch.inference_mode()
def grounding_with_visual_prompt_and_attention(
    processor, 
    image, 
    visual_prompt_embed, 
    visual_prompt_mask, 
    text_prompt="visual"
):
    """
    使用 Visual Prompt 在 Query Image 上進行 Grounding，並返回 attention maps
    
    Returns:
        state: 包含 masks, boxes, scores 的狀態
        attention_info: 包含 encoder 和 decoder attention 的資訊
    """
    from examples.sam3_utils import box_cxcywh_to_xyxy
    from torch.nn.functional import interpolate
    
    attention_info = {}
    
    # 設定 Query Image
    state = processor.set_image(image)
    
    # 設定 Text Prompt
    text_outputs = processor.model.backbone.forward_text([text_prompt], device=processor.device)
    state["backbone_out"].update(text_outputs)
    
    # 準備輸入
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
        
    backbone_out = state["backbone_out"]
    find_input = processor.find_stage
    geometric_prompt = state["geometric_prompt"]
    
    # =========================================================================
    # Step 1: _encode_prompt (注入 visual_prompt_embed)
    # =========================================================================
    prompt, prompt_mask, backbone_out = processor.model._encode_prompt(
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=visual_prompt_embed,
        visual_prompt_mask=visual_prompt_mask
    )
    
    # 保存 prompt 資訊
    attention_info['prompt'] = prompt.clone()
    attention_info['prompt_mask'] = prompt_mask.clone()
    
    # =========================================================================
    # Step 2: _run_encoder - 這裡有 cross-attention
    # =========================================================================
    # 獲取 encoder 的輸入以便後續計算 attention
    feat_tuple = processor.model._get_img_feats(backbone_out, find_input.img_ids)
    _, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
    
    attention_info['img_feats'] = [f.clone() for f in img_feats]
    attention_info['vis_feat_sizes'] = vis_feat_sizes
    
    backbone_out, encoder_out, _ = processor.model._run_encoder(
        backbone_out, find_input, prompt, prompt_mask
    )
    
    attention_info['encoder_hidden_states'] = encoder_out["encoder_hidden_states"].clone()
    attention_info['spatial_shapes'] = encoder_out.get("spatial_shapes")
    
    out = {
        "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        "prev_encoder_out": {
            "encoder_out": encoder_out,
            "backbone_out": backbone_out,
        },
    }
    
    # =========================================================================
    # Step 3: _run_decoder - 這裡有 query 到 image 的 cross-attention
    # =========================================================================
    out, hs = processor.model._run_decoder(
        memory=out["encoder_hidden_states"],
        pos_embed=encoder_out["pos_embed"],
        src_mask=encoder_out["padding_mask"],
        out=out,
        prompt=prompt,
        prompt_mask=prompt_mask,
        encoder_out=encoder_out,
    )
    
    # 保存 decoder hidden states
    attention_info['decoder_hs'] = hs.clone()
    
    # =========================================================================
    # Step 4: _run_segmentation_heads
    # =========================================================================
    processor.model._run_segmentation_heads(
        out=out,
        backbone_out=backbone_out,
        img_ids=find_input.img_ids,
        vis_feat_sizes=encoder_out["vis_feat_sizes"],
        encoder_hidden_states=out["encoder_hidden_states"],
        prompt=prompt,
        prompt_mask=prompt_mask,
        hs=hs,
    )
    
    # 後處理
    out_bbox = out["pred_boxes"]
    out_logits = out["pred_logits"]
    out_masks = out["pred_masks"]
    out_probs = out_logits.sigmoid()
    presence_score = out["presence_logit_dec"].sigmoid().unsqueeze(1)
    out_probs = (out_probs * presence_score).squeeze(-1)

    keep = out_probs > processor.confidence_threshold
    out_probs = out_probs[keep]
    out_masks = out_masks[keep]
    out_bbox = out_bbox[keep]

    boxes = box_cxcywh_to_xyxy(out_bbox)

    img_h = state["original_height"]
    img_w = state["original_width"]
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(processor.device)
    boxes = boxes * scale_fct[None, :]

    out_masks = interpolate(
        out_masks.unsqueeze(1),
        (img_h, img_w),
        mode="bilinear",
        align_corners=False,
    ).sigmoid()

    state["masks_logits"] = out_masks.detach()
    state["masks"] = (out_masks > 0.5).detach()
    state["boxes"] = boxes.detach()
    state["scores"] = out_probs.detach()
    
    return state, attention_info


# ============================================================================
# Attention 視覺化函數
# ============================================================================

def visualize_prompt_attention_on_image(
    attention_info,
    query_image,
    support_box=None,
    title="Visual Prompt Attention on Query Image"
):
    """
    視覺化 visual prompt 在 query image 上的 attention
    
    策略: 使用 encoder hidden states 來計算 prompt 到 image 的相關性
    """
    # 獲取 encoder 輸出和 prompt
    encoder_hidden = attention_info['encoder_hidden_states']  # (HW, B, C)
    prompt = attention_info['prompt']  # (N_prompt, B, C)
    spatial_shapes = attention_info.get('spatial_shapes')
    
    # 計算 prompt 到 encoder hidden states 的相似度
    # prompt: (N, B, C), encoder_hidden: (HW, B, C)
    
    # 取 batch=0
    prompt_feat = prompt[:, 0, :]  # (N_prompt, C)
    encoder_feat = encoder_hidden[:, 0, :]  # (HW, C)
    
    # 計算 cosine similarity
    prompt_feat_norm = F.normalize(prompt_feat, dim=-1)
    encoder_feat_norm = F.normalize(encoder_feat, dim=-1)
    
    # 對每個 prompt token，計算與所有 image positions 的相似度
    similarity = torch.matmul(prompt_feat_norm, encoder_feat_norm.T)  # (N_prompt, HW)
    
    # 取平均或取 visual prompt token 的 similarity
    # 通常 visual prompt 是 prompt 中的特定部分
    avg_similarity = similarity.mean(dim=0)  # (HW,)
    
    # Reshape 到空間維度
    if spatial_shapes is not None:
        H, W = spatial_shapes[0].tolist()
    else:
        # 假設是正方形
        HW = avg_similarity.shape[0]
        H = W = int(HW ** 0.5)
    
    attention_map = avg_similarity.view(H, W).cpu().numpy()
    
    # 正規化
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # 視覺化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始 Query Image
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis('off')
    
    # Attention Map
    im = axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title("Attention Map (Feature Space)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    # Resize attention map to image size
    attention_resized = np.array(Image.fromarray((attention_map * 255).astype(np.uint8)).resize(
        query_image.size, Image.BILINEAR
    )) / 255.0
    
    axes[2].imshow(query_image)
    axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
    axes[2].set_title("Attention Overlay")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return attention_map


def visualize_detection_attention(
    attention_info,
    query_image,
    detected_boxes,
    title="Detection Query Attention"
):
    """
    視覺化偵測 query 對 image 的 attention
    使用 decoder hidden states
    """
    decoder_hs = attention_info['decoder_hs']  # (num_layers, num_queries, B, C)
    encoder_hidden = attention_info['encoder_hidden_states']  # (HW, B, C)
    spatial_shapes = attention_info.get('spatial_shapes')
    
    # 取最後一層的 decoder hidden states
    last_layer_hs = decoder_hs[-1, :, 0, :]  # (num_queries, C)
    encoder_feat = encoder_hidden[:, 0, :]  # (HW, C)
    
    # 正規化
    query_norm = F.normalize(last_layer_hs, dim=-1)
    encoder_norm = F.normalize(encoder_feat, dim=-1)
    
    # 對每個 detection query 計算 attention
    query_to_img_sim = torch.matmul(query_norm, encoder_norm.T)  # (num_queries, HW)
    
    if spatial_shapes is not None:
        H, W = spatial_shapes[0].tolist()
    else:
        HW = encoder_feat.shape[0]
        H = W = int(HW ** 0.5)
    
    # 找到高分的 queries (對應到被偵測到的物體)
    num_detected = len(detected_boxes)
    
    if num_detected == 0:
        print("No detections to visualize")
        return None
    
    fig, axes = plt.subplots(1, min(num_detected + 1, 5), figsize=(5 * min(num_detected + 1, 5), 5))
    if num_detected == 0:
        axes = [axes]
    
    # 原始圖片
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image with Detections")
    for box in detected_boxes.cpu().numpy():
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        axes[0].add_patch(rect)
    axes[0].axis('off')
    
    # 每個偵測的 attention
    for i in range(min(num_detected, 4)):
        attn_map = query_to_img_sim[i].view(H, W).cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        attn_resized = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            query_image.size, Image.BILINEAR
        )) / 255.0
        
        axes[i + 1].imshow(query_image)
        axes[i + 1].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[i + 1].set_title(f"Detection {i+1} Attention")
        axes[i + 1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return query_to_img_sim


# ============================================================================
# 使用範例
# ============================================================================

"""
使用方法:

1. 載入模型和圖片 (參考原始 notebook)
2. 提取 visual prompt embedding
3. 使用修改版函數獲取 attention 資訊
4. 視覺化

# 範例程式碼:
from sam3_utils import *
from visualize_attention import *

# 載入模型
model = build_sam3_image_model(checkpoint="sam3.pt")
processor = Sam3Processor(model, device="cuda")

# 載入圖片
support_image = Image.open("support.jpg")
query_image = Image.open("query.jpg")
support_box = [x, y, w, h]

# 提取 visual prompt
visual_prompt_embed, visual_prompt_mask = extract_visual_prompt_embedding(
    processor, support_image, support_box
)

# 執行偵測並獲取 attention
state, attention_info = grounding_with_visual_prompt_and_attention(
    processor, query_image, visual_prompt_embed, visual_prompt_mask
)

# 視覺化 prompt attention
attention_map = visualize_prompt_attention_on_image(
    attention_info, query_image, support_box
)

# 視覺化 detection attention
if len(state["boxes"]) > 0:
    visualize_detection_attention(
        attention_info, query_image, state["boxes"]
    )
"""
