import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh, box_cxcywh_to_xyxy

import sam3

def load_sam3_model(checkpoint_path, device="cuda"):
    """
    Loads the SAM3 model and processor.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    
    # Enable TF32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        bpe_path=bpe_path
    )
    processor = Sam3Processor(model, confidence_threshold=0.4)
    return processor

@torch.inference_mode()
def extract_visual_prompt_embedding(processor, image, box):
    """
    Extracts Visual Prompt Embedding from Support Image BBox.
    
    Args:
        processor: Sam3Processor instance
        image: PIL Image, Support Image
        box: List [x, y, w, h] or similar, Bounding Box
    
    Returns:
        visual_prompt_embed: Extracted feature Tensor
    """
    # Set image
    state = processor.set_image(image)
    
    # Normalize BBox (x, y, w, h) -> (cx, cy, w, h) normalized
    width, height = image.size
    box_val = torch.tensor(box, dtype=torch.float32).view(-1, 4)
    box_cxcywh = box_xywh_to_cxcywh(box_val)
    box_cxcywh_norm = box_cxcywh / torch.tensor([width, height, width, height])
    
    # Add Geometric Prompt
    if "language_features" not in state["backbone_out"]:
        dummy_text_outputs = processor.model.backbone.forward_text(["visual"], device=processor.device)
        state["backbone_out"].update(dummy_text_outputs)
        
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
        
    # Append Box Prompt
    boxes = box_cxcywh_norm.to(processor.device).view(1, 1, 4)
    labels = torch.tensor([True], device=processor.device, dtype=torch.bool).view(1, 1)
    state["geometric_prompt"].append_boxes(boxes, labels)
    
    # Encode prompt manually to get features
    find_input = processor.find_stage
    geometric_prompt = state["geometric_prompt"]
    backbone_out = state["backbone_out"]
    
    # Get image feats
    feat_tuple = processor.model._get_img_feats(backbone_out, find_input.img_ids)
    _, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
    
    # Encode geometry
    geo_feats, geo_masks = processor.model.geometry_encoder(
        geo_prompt=geometric_prompt,
        img_feats=img_feats,
        img_sizes=vis_feat_sizes,
        img_pos_embeds=img_pos_embeds,
    )
    
    # geo_feats is the visual prompt embedding we need
    return geo_feats

@torch.inference_mode()
def grounding_with_visual_prompt(processor, image, visual_prompt_embed):
    """
    Performs grounding on Query Image using Visual Prompt Embedding.
    
    Args:
        processor: Sam3Processor instance
        image: PIL Image, Query Image
        visual_prompt_embed: Tensor, from extract_visual_prompt_embedding
        
    Returns:
        state: Dict containing 'boxes', 'scores', etc.
    """
    # Set image
    state = processor.set_image(image)
    
    # Prepare dummy prompts
    if "language_features" not in state["backbone_out"]:
        dummy_text_outputs = processor.model.backbone.forward_text(["visual"], device=processor.device)
        state["backbone_out"].update(dummy_text_outputs)
        
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
        
    # Get inputs
    find_input = processor.find_stage
    find_target = None 
    geometric_prompt = state["geometric_prompt"]
    backbone_out = state["backbone_out"]
    
    # 1. Encode Prompt (Inject Visual Prompt)
    prompt, prompt_mask, backbone_out = processor.model._encode_prompt(
        backbone_out, find_input, geometric_prompt,
        visual_prompt_embed=visual_prompt_embed, # Inject!
        visual_prompt_mask=torch.zeros((1, visual_prompt_embed.shape[0]), device=processor.device, dtype=torch.bool) # Correct mask shape
    )
    
    # 2. Run Encoder
    backbone_out, encoder_out, _ = processor.model._run_encoder(
        backbone_out, find_input, prompt, prompt_mask
    )
    
    out = {
        "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        "prev_encoder_out": {
            "encoder_out": encoder_out,
            "backbone_out": backbone_out,
        },
    }
    
    # 3. Run Decoder
    out, hs = processor.model._run_decoder(
        memory=out["encoder_hidden_states"],
        pos_embed=encoder_out["pos_embed"],
        src_mask=encoder_out["padding_mask"],
        out=out,
        prompt=prompt,
        prompt_mask=prompt_mask,
        encoder_out=encoder_out,
    )
    
    # 4. Post-process (Get boxes)
    # We only need the boxes and scores
    pred_logits = out["pred_logits"]
    pred_boxes = out["pred_boxes"]
    
    prob = pred_logits.sigmoid()
    scores, labels = prob.max(-1)
    
    # Un-normalize boxes
    width, height = image.size
    boxes_cxcywh = pred_boxes * torch.tensor([width, height, width, height], device=processor.device)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    
    state["boxes"] = boxes_xyxy[0].detach().cpu().numpy() # [N, 4]
    state["scores"] = scores[0].detach().cpu().numpy()    # [N]
    
    return state

def filter_results(state, threshold=0.5):
    """Filters boxes by score threshold."""
    boxes = state["boxes"]
    scores = state["scores"]
    keep = scores > threshold
    return boxes[keep], scores[keep]

def draw_boxes_pil(image, boxes, scores=None, color="red", width=3):
    """
    Draws boxes on a PIL image.
    boxes: [N, 4] (x1, y1, x2, y2)
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a larger font
    try:
        # Windows usually has arial.ttf
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        # Fallback to default
        font = None # Use default

    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if scores is not None:
            score = scores[i]
            # Draw score
            # A simple text drawing, might be small without a ttf font
            # but default font is usually available
            text = f"{score:.2f}"
            
            # Draw text background
            # Get text bounding box is tricky without font metrics, try simple approach
            # Using defaults
            
            # Simple text top-left corner
            text_pos = (x1, max(0, y1 - 40)) # Adjust position for larger font
            if font:
                draw.text(text_pos, text, fill=color, font=font)
            else:
                draw.text(text_pos, text, fill=color)
            
    return img_copy


@torch.inference_mode()
def grounding_with_visual_prompt_and_attention(processor, image, visual_prompt_embed):
    """
    Performs grounding on Query Image and returns both detection results and attention info.
    
    Args:
        processor: Sam3Processor instance
        image: PIL Image, Query Image
        visual_prompt_embed: Tensor, from extract_visual_prompt_embedding
        
    Returns:
        state: Dict containing 'boxes', 'scores', etc.
        attention_info: Dict containing encoder/decoder hidden states for visualization
    """
    attention_info = {}
    
    # Set image
    state = processor.set_image(image)
    
    # Prepare dummy prompts
    if "language_features" not in state["backbone_out"]:
        dummy_text_outputs = processor.model.backbone.forward_text(["visual"], device=processor.device)
        state["backbone_out"].update(dummy_text_outputs)
        
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
        
    # Get inputs
    find_input = processor.find_stage
    geometric_prompt = state["geometric_prompt"]
    backbone_out = state["backbone_out"]
    
    # Create mask for visual prompt
    visual_prompt_mask = torch.zeros(
        (1, visual_prompt_embed.shape[0]), 
        device=processor.device, 
        dtype=torch.bool
    )
    
    # 1. Encode Prompt (Inject Visual Prompt)
    prompt, prompt_mask, backbone_out = processor.model._encode_prompt(
        backbone_out, find_input, geometric_prompt,
        visual_prompt_embed=visual_prompt_embed,
        visual_prompt_mask=visual_prompt_mask
    )
    
    # Save prompt info
    attention_info['prompt'] = prompt.clone()
    attention_info['prompt_mask'] = prompt_mask.clone()
    
    # 2. Run Encoder
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
    
    # 3. Run Decoder
    out, hs = processor.model._run_decoder(
        memory=out["encoder_hidden_states"],
        pos_embed=encoder_out["pos_embed"],
        src_mask=encoder_out["padding_mask"],
        out=out,
        prompt=prompt,
        prompt_mask=prompt_mask,
        encoder_out=encoder_out,
    )
    
    attention_info['decoder_hs'] = hs.clone()
    
    # 4. Post-process (Get boxes)
    pred_logits = out["pred_logits"]
    pred_boxes = out["pred_boxes"]
    
    prob = pred_logits.sigmoid()
    scores, labels = prob.max(-1)
    
    # Un-normalize boxes
    width, height = image.size
    boxes_cxcywh = pred_boxes * torch.tensor([width, height, width, height], device=processor.device)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    
    state["boxes"] = boxes_xyxy[0].detach().cpu().numpy()
    state["scores"] = scores[0].detach().cpu().numpy()
    
    return state, attention_info


def create_attention_overlay(attention_info, query_image):
    """
    Creates an attention heatmap overlay showing how the visual prompt attends to the query image.
    
    The attention is computed as the similarity between the visual prompt features
    and the encoder features from the query image. This shows which regions of the
    query image are most similar to the support image's bounding box region.
    
    Args:
        attention_info: Dict from grounding_with_visual_prompt_and_attention
        query_image: PIL Image, Query Image
        
    Returns:
        overlay_image: PIL Image with attention heatmap overlay
        attention_map: numpy array of the raw attention map
    """
    import torch.nn.functional as F
    
    # Get encoder output (this is specific to each query image)
    encoder_hidden = attention_info['encoder_hidden_states']  # (HW, B, C)
    # Get prompt (this is the visual prompt from support image)
    prompt = attention_info['prompt']  # (N_prompt, B, C)
    spatial_shapes = attention_info.get('spatial_shapes')
    
    # Take batch=0
    prompt_feat = prompt[:, 0, :]  # (N_prompt, C)
    encoder_feat = encoder_hidden[:, 0, :]  # (HW, C) - this is from query image
    
    # Compute cosine similarity between prompt and encoder features
    prompt_feat_norm = F.normalize(prompt_feat.float(), dim=-1)
    encoder_feat_norm = F.normalize(encoder_feat.float(), dim=-1)
    
    # Average prompt features then compute similarity
    avg_prompt = prompt_feat_norm.mean(dim=0, keepdim=True)  # (1, C)
    
    # Compute similarity: how similar is each image position to the prompt?
    similarity = torch.matmul(encoder_feat_norm, avg_prompt.T).squeeze(-1)  # (HW,)
    
    # Reshape to spatial dimensions
    if spatial_shapes is not None:
        H, W = spatial_shapes[0].tolist()
    else:
        HW = similarity.shape[0]
        H = W = int(HW ** 0.5)
    
    attention_map = similarity.view(H, W).cpu().numpy()
    
    # Normalize to 0-1 range
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Resize to image size
    import matplotlib.cm as cm
    
    # Create heatmap using jet colormap
    attention_resized = np.array(Image.fromarray(
        (attention_map * 255).astype(np.uint8)
    ).resize(query_image.size, Image.BILINEAR)) / 255.0
    
    # Apply colormap
    heatmap = cm.jet(attention_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap)
    
    # Blend with original image
    overlay = Image.blend(query_image.convert("RGB"), heatmap_pil, alpha=0.4)
    
    return overlay, attention_map

