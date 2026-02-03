
import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
import os
from sam3 import sam3_model_registry, Sam3Processor
from sam3.model.data_misc import FindStage
from sam3.model import box_ops
from streamlit_drawable_canvas import st_canvas
import matplotlib.cm as cm

# Page Config
st.set_page_config(layout="wide", page_title="SAM3 Few-Shot Object Detection")

# Setup Sidebar
st.sidebar.title("Configuration")
checkpoint_path = st.sidebar.text_input("Checkpoint Path", os.path.abspath("sam3.pt"))
model_type = st.sidebar.selectbox("Model Type", ["vit_l", "vit_b", "vit_h"], index=0)
device = st.sidebar.selectbox("Device", ["cuda", "cpu", "mps"], index=0)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)

@st.cache_resource
def load_model(checkpoint, model_type, device):
    if not os.path.exists(checkpoint):
        return None
    try:
        sam3 = sam3_model_registry[model_type](checkpoint=checkpoint)
        sam3.to(device=device)
        sam3.eval()
        processor = Sam3Processor(sam3, device=device)
        return sam3, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def run_inference(sam3, processor, ref_image, target_images, ref_box_norm, device):
    """
    Run SAM3 inference with a reference box prompt on a batch of target images.
    """
    # Prepare batch: [Ref, Target1, Target2, ...]
    images = [ref_image] + target_images
    
    # Process batch
    try:
        state = processor.set_image_batch(images)
    except Exception as e:
        st.error(f"Error checking images: {e}")
        return None

    # Construct Prompt Input (FindStage)
    # img_ids=[0] -> Prompt is on the first image (Reference)
    # ref_box_norm is [cx, cy, w, h] normalized
    find_inputs = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=torch.tensor([ref_box_norm], device=device, dtype=torch.float32).view(1, 1, 4),
        input_boxes_mask=torch.tensor([[True]], device=device, dtype=torch.bool),
        input_boxes_label=torch.tensor([[True]], device=device, dtype=torch.long),
        input_points=None,
        input_points_mask=None,
    )
    
    geometric_prompt = sam3._get_dummy_prompt()
    
    with torch.inference_mode():
        outputs = sam3.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_inputs,
            geometric_prompt=geometric_prompt,
            find_target=None 
        )
    
    # Process results for target images (indices 1 to N)
    results = []
    
    # Check outputs keys
    for i, target_img in enumerate(target_images):
        target_idx = i + 1 # 0 is ref
        
        out_bbox = outputs["pred_boxes"][target_idx]
        out_logits = outputs["pred_logits"][target_idx]
        
        # Scores
        out_probs = out_logits.sigmoid()
        if "presence_logit_dec" in outputs:
             presence_score = outputs["presence_logit_dec"][target_idx].sigmoid()
             out_probs = out_probs * presence_score
        
        out_probs = out_probs.squeeze(-1)
        
        # Filter
        keep = out_probs > confidence_threshold
        final_probs = out_probs[keep]
        final_boxes_norm = out_bbox[keep] # [cx, cy, w, h]
        
        # Convert to XYXY for drawing
        final_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(final_boxes_norm)
        
        # Scale to original image size
        w, h = target_img.size
        # Use simple tensor scale
        scale_fct = torch.tensor([w, h, w, h], device=device)
        final_boxes_abs = final_boxes_xyxy * scale_fct[None, :]
        
        results.append({
            "boxes": final_boxes_abs.cpu().numpy(),
            "scores": final_probs.cpu().numpy(),
            "image": target_img
        })
        
    return results

# Main UI
st.title("SAM3 Cross-Image Detection")

# Load Model
model_load = load_model(checkpoint_path, model_type, device)

if model_load:
    sam3, processor = model_load
    st.success(f"Model loaded from {checkpoint_path} on {device}")
    
    # 1. Reference Image
    st.header("1. Reference Image")
    ref_file = st.file_uploader("Upload Reference Image", type=["jpg", "png", "jpeg"], key="ref")
    
    ref_image = None
    ref_box_norm = None
    
    if ref_file:
        ref_image = Image.open(ref_file).convert("RGB")
        w, h = ref_image.size
        
        # Canvas for drawing bbox
        display_width = 600
        scale_factor = 1.0
        display_height = h
        
        if w > display_width:
            display_height = int(h * (display_width / w))
            resized_ref = ref_image.resize((display_width, display_height))
            scale_factor = w / display_width
        else:
            resized_ref = ref_image
            display_width = w
            display_height = h

        st.write("Draw a box around the object you want to detect:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=resized_ref,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                box_obj = objects[-1] 
                left = box_obj["left"] * scale_factor
                top = box_obj["top"] * scale_factor
                width = box_obj["width"] * scale_factor
                height = box_obj["height"] * scale_factor
                
                x0, y0 = left, top
                x1, y1 = left + width, top + height
                
                cx = (x0 + x1) / 2 / w
                cy = (y0 + y1) / 2 / h
                nw = width / w
                nh = height / h
                
                ref_box_norm = [cx, cy, nw, nh]
                st.info(f"Reference Box Selected: [x={x0:.1f}, y={y0:.1f}, w={width:.1f}, h={height:.1f}]")

    # 2. Target Images
    st.header("2. Target Images")
    target_files = st.file_uploader("Upload Target Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="targets")

    # Main Control Logic
    if ref_image and ref_box_norm and target_files:
        
        if st.button("Run Detection"):
            target_images_loaded = [Image.open(f).convert("RGB") for f in target_files]
            
            with st.spinner("Running SAM3 Inference..."):
                processor.confidence_threshold = confidence_threshold
                results = run_inference(sam3, processor, ref_image, target_images_loaded, ref_box_norm, device)
                
            if results:
                st.session_state['detection_results'] = results
                st.rerun()
            else:
                st.error("Detection failed or no objects found.")
        
        # Display Results if available
        if 'detection_results' in st.session_state:
            st.header("Results")
            results = st.session_state['detection_results']
            cols = st.columns(min(len(results), 3))
            
            for idx, res in enumerate(results):
                img = res["image"].copy()
                draw = ImageDraw.Draw(img)
                boxes = res["boxes"]
                scores = res["scores"]
                
                for box, score in zip(boxes, scores):
                    draw.rectangle(list(box), outline="green", width=4)
                    draw.text((box[0], box[1]), f"{score:.2f}", fill="white")
                
                with cols[idx % 3]:
                    st.image(img, caption=f"Target {idx+1} ({len(boxes)} objects)", use_column_width=True)

            # Feature Analysis Section
            st.divider()
            st.header("Feature Space Analysis")
            
            vis_type = st.radio("Visualization Type", ["Cosine Similarity Heatmap", "Principal Component Analysis (PCA)"], horizontal=True)
            
            if st.button("Generate Visualization"):
                # Reload target images
                target_images = [Image.open(f).convert("RGB") for f in target_files]
                images = [ref_image] + target_images
                
                with st.spinner(f"Computing {vis_type}..."):
                    state = processor.set_image_batch(images)
                    
                    # Last layer features: [Batch, C, H, W]
                    features = state["backbone_out"]["backbone_fpn"][-1]
                    B, C, Hf, Wf = features.shape

                    # Ref Box Coords on Feature Map
                    cx, cy, w, h = ref_box_norm
                    x0 = int((cx - w/2) * Wf)
                    y0 = int((cy - h/2) * Hf)
                    x1 = int((cx + w/2) * Wf)
                    y1 = int((cy + h/2) * Hf)
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(Wf, x1), min(Hf, y1)

                    if x1 <= x0 or y1 <= y0:
                        st.error("Reference box too small/invalid on feature map.")
                    else:
                        vis_cols = st.columns(min(len(images), 3))
                        
                        if vis_type == "Cosine Similarity Heatmap":
                            # Prototype Vector
                            # Use Reference Image (Index 0) features
                            ref_feat_map = features[0] 
                            ref_crop = ref_feat_map[:, y0:y1, x0:x1]
                            ref_vector = ref_crop.mean(dim=(1, 2))
                            ref_vector = torch.nn.functional.normalize(ref_vector, dim=0)

                            for i in range(B):
                                feat_map = features[i]
                                feat_norm = torch.nn.functional.normalize(feat_map, dim=0)
                                # Cos Sim
                                sim_map = torch.einsum("c,chw->hw", ref_vector, feat_norm)
                                sim_map_np = sim_map.cpu().numpy()
                                
                                # Scale 0-1
                                sim_map_np = np.maximum(0, sim_map_np)
                                sim_map_np = (sim_map_np - sim_map_np.min()) / (sim_map_np.max() - sim_map_np.min() + 1e-6)
                                
                                # Heatmap
                                heatmap = cm.jet(sim_map_np)[:, :, :3]
                                heatmap = (heatmap * 255).astype(np.uint8)
                                
                                # Overlay
                                orig_w, orig_h = images[i].size
                                heatmap_img = Image.fromarray(heatmap).resize((orig_w, orig_h), Image.BILINEAR)
                                
                                original = images[i].convert("RGBA")
                                heatmap_rgba = heatmap_img.convert("RGBA")
                                heatmap_rgba.putalpha(160)
                                
                                blended = Image.alpha_composite(original, heatmap_rgba)
                                title = "Reference" if i == 0 else f"Target {i}"
                                with vis_cols[i % 3]:
                                    st.image(blended, caption=title, use_column_width=True)

                        elif vis_type == "Principal Component Analysis (PCA)":
                            # Reshape to [N, C]
                            flat_features = features.permute(0, 2, 3, 1).reshape(-1, C)
                            flat_features = torch.nn.functional.normalize(flat_features, dim=1)
                            
                            # PCA
                            mean = flat_features.mean(dim=0)
                            centered = flat_features - mean
                            U, S, V = torch.pca_lowrank(centered, q=3)
                            projected = torch.matmul(centered, V[:, :3])
                            
                            # Norm 0-1
                            p_min = projected.min(dim=0)[0]
                            p_max = projected.max(dim=0)[0]
                            projected = (projected - p_min) / (p_max - p_min + 1e-6)
                            
                            projected = projected.reshape(B, Hf, Wf, 3)
                            projected_np = (projected.cpu().numpy() * 255).astype(np.uint8)
                            
                            for i in range(B):
                                pca_img_small = Image.fromarray(projected_np[i])
                                orig_w, orig_h = images[i].size
                                pca_img = pca_img_small.resize((orig_w, orig_h), Image.NEAREST)
                                title = "Reference" if i == 0 else f"Target {i}"
                                with vis_cols[i % 3]:
                                    st.image(pca_img, caption=title, use_column_width=True)

    elif not ref_image:
        st.warning("Please upload a reference image.")
    elif not ref_box_norm:
        st.warning("Please draw a bounding box on the reference image.")
    elif not target_files:
        st.warning("Please upload at least one target image.")

else:
    st.warning(f"Please check the checkpoint path on the sidebar.")
