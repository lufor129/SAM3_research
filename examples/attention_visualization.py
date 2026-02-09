"""
SAM3 Cross-Image Few-Shot Detection - Attention 視覺化模組

此模組提供將 Visual Prompt Attention 可視化的功能，
用於解釋 Support Image 的 BBox 區域在 Query Image 上關注了哪些位置。

使用方法：
    1. 在 notebook 中 import 此模組
    2. 使用 grounding_with_attention_output() 取代標準的 grounding_with_visual_prompt()
    3. 調用視覺化函數來查看 attention maps

範例：
    from attention_visualization import (
        grounding_with_attention_output,
        visualize_prompt_attention,
        visualize_detection_query_attention
    )
    
    # 執行偵測並獲取 attention
    state, attention_info = grounding_with_attention_output(
        processor, query_image, visual_prompt_embed, visual_prompt_mask
    )
    
    # 視覺化 attention
    visualize_prompt_attention(attention_info, query_image)
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import interpolate


@torch.inference_mode()
def grounding_with_attention_output(processor, image, visual_prompt_embed, visual_prompt_mask, text_prompt="visual"):
    """
    使用 Visual Prompt 在 Query Image 上進行 Grounding，並返回 attention 相關資訊。
    
    Args:
        processor: Sam3Processor 實例
        image: PIL Image, Query Image
        visual_prompt_embed: 來自 Support Image 的特徵
        visual_prompt_mask: 來自 Support Image 的遮罩
        text_prompt: 文字提示
        
    Returns:
        state: 包含 masks, boxes, scores 的狀態
        attention_info: 包含 encoder 和 decoder 輸出的資訊，用於視覺化
    """
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
    
    # Step 1: _encode_prompt (注入 visual_prompt_embed)
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
    
    # Step 2: _run_encoder
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
    
    # Step 3: _run_decoder
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
    
    # Step 4: _run_segmentation_heads
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


def visualize_prompt_attention(attention_info, query_image, title="Visual Prompt Attention", save_path=None):
    """
    視覺化 Visual Prompt 對 Query Image 的 Attention 分佈。
    
    透過計算 prompt embedding 與 encoder hidden states 的 cosine similarity，
    產生 attention map 來顯示 Support BBox 區域在 Query Image 上「關注」的位置。
    
    Args:
        attention_info: grounding_with_attention_output 返回的 attention 資訊
        query_image: PIL Image, Query Image
        title: 圖表標題
        save_path: 若指定，則儲存圖片到該路徑
        
    Returns:
        attention_map: numpy array, attention 熱圖
    """
    # 獲取 encoder 輸出和 prompt
    encoder_hidden = attention_info['encoder_hidden_states']  # (HW, B, C)
    prompt = attention_info['prompt']  # (N_prompt, B, C)
    spatial_shapes = attention_info.get('spatial_shapes')
    
    # 取 batch=0
    prompt_feat = prompt[:, 0, :]  # (N_prompt, C)
    encoder_feat = encoder_hidden[:, 0, :]  # (HW, C)
    
    # 計算 cosine similarity
    prompt_feat_norm = F.normalize(prompt_feat, dim=-1)
    encoder_feat_norm = F.normalize(encoder_feat, dim=-1)
    
    # 對每個 prompt token，計算與所有 image positions 的相似度
    similarity = torch.matmul(prompt_feat_norm, encoder_feat_norm.T)  # (N_prompt, HW)
    avg_similarity = similarity.mean(dim=0)  # (HW,)
    
    # Reshape 到空間維度
    if spatial_shapes is not None:
        H, W = spatial_shapes[0].tolist()
    else:
        HW = avg_similarity.shape[0]
        H = W = int(HW ** 0.5)
    
    attention_map = avg_similarity.view(H, W).float().cpu().numpy()
    
    # 正規化
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Resize to image size
    attention_resized = np.array(Image.fromarray((attention_map * 255).astype(np.uint8)).resize(
        query_image.size, Image.BILINEAR
    )) / 255.0
    
    # 視覺化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis('off')
    
    im = axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title("Attention Map (Feature Space)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    axes[2].imshow(query_image)
    axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
    axes[2].set_title("Attention Overlay")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖片已儲存至: {save_path}")
    
    plt.show()
    
    return attention_map


def visualize_detection_query_attention(attention_info, query_image, detected_boxes, save_path=None):
    """
    視覺化每個偵測 query 對 image 的 attention。
    
    這顯示了模型在進行偵測時，每個 detection query 關注 image 的哪些區域。
    
    Args:
        attention_info: grounding_with_attention_output 返回的 attention 資訊
        query_image: PIL Image, Query Image
        detected_boxes: Tensor, 偵測到的 bounding boxes
        save_path: 若指定，則儲存圖片到該路徑
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
    
    # 計算 attention
    query_to_img_sim = torch.matmul(query_norm, encoder_norm.T).float()  # (num_queries, HW)
    
    if spatial_shapes is not None:
        H, W = spatial_shapes[0].tolist()
    else:
        HW = encoder_feat.shape[0]
        H = W = int(HW ** 0.5)
    
    num_detected = len(detected_boxes)
    num_queries = query_to_img_sim.shape[0]
    
    if num_detected == 0:
        print("沒有偵測結果可視覺化")
        return
    
    # 限制顯示數量為實際可用的 query 數量和偵測數量的最小值
    num_to_show = min(num_detected, num_queries, 4)
    fig, axes = plt.subplots(1, num_to_show + 1, figsize=(5 * (num_to_show + 1), 5))
    
    # 確保 axes 是 array
    if num_to_show == 0:
        axes = [axes]
    
    # 原始圖片與偵測框
    axes[0].imshow(query_image)
    axes[0].set_title("Detections")
    for i, box in enumerate(detected_boxes.cpu().numpy()[:num_to_show]):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                              color=plt.cm.tab10(i), linewidth=2)
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-5, f'Det {i+1}', color=plt.cm.tab10(i), fontsize=10)
    axes[0].axis('off')
    
    # 每個偵測的 attention
    for i in range(num_to_show):
        attn_map = query_to_img_sim[i].view(H, W).cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        attn_resized = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            query_image.size, Image.BILINEAR
        )) / 255.0
        
        axes[i + 1].imshow(query_image)
        axes[i + 1].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[i + 1].set_title(f"Detection {i+1} Attention")
        axes[i + 1].axis('off')
    
    plt.suptitle("Detection Query Attention Maps")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖片已儲存至: {save_path}")
    
    plt.show()


# ============================================================================
# 完整使用範例（可直接在 notebook 中執行）
# ============================================================================
EXAMPLE_CODE = '''
# ============================================================================
# 步驟 1: 載入此模組
# ============================================================================
from attention_visualization import (
    grounding_with_attention_output,
    visualize_prompt_attention,
    visualize_detection_query_attention
)

# ============================================================================
# 步驟 2: 執行帶 Attention 輸出的偵測
# ============================================================================
# 假設您已執行過 extract_visual_prompt_embedding() 獲得:
# - visual_prompt_embed
# - visual_prompt_mask

state_with_attn, attention_info = grounding_with_attention_output(
    processor, 
    query_image,          # Query Image (PIL Image)
    visual_prompt_embed,   # 從 Support Image 提取的特徵
    visual_prompt_mask     # 對應的 mask
)

# ============================================================================
# 步驟 3: 視覺化 Visual Prompt Attention
# ============================================================================
# 這顯示 Support Image 的 BBox 區域在 Query Image 上「關注」的位置
attention_map = visualize_prompt_attention(
    attention_info, 
    query_image,
    title="Support BBox → Query Image Attention"
)

# ============================================================================
# 步驟 4: 視覺化 Detection Query Attention
# ============================================================================
# 這顯示每個偵測 query 在 image 上關注的區域
if len(state_with_attn['boxes']) > 0:
    visualize_detection_query_attention(
        attention_info, 
        query_image, 
        state_with_attn['boxes']
    )
'''

if __name__ == "__main__":
    print("SAM3 Attention 視覺化模組已載入！")
    print("\n使用方法：")
    print(EXAMPLE_CODE)
