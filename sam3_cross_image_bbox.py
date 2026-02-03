
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from sam3 import sam3_model_registry, Sam3Processor
from sam3.model.data_misc import FindStage
from sam3.model import box_ops
from torchvision.transforms import v2

def show_results(image, boxes, scores, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    for box, score in zip(boxes, scores):
        if score < 0.5: continue # Threshold
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', linewidth=2))
        ax.text(x0, y0, f"{score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
        
    plt.axis('off')
    plt.savefig(output_path)
    print(f"Saved result to {output_path}")
    plt.close()

def run_cross_image_detection(
    ref_image_path, 
    target_image_path, 
    ref_box, 
    checkpoint, 
    model_type="vit_l",
    device="cuda"
):
    """
    Run SAM3 cross-image few-shot detection.
    
    Args:
        ref_image_path: Path to the reference image.
        target_image_path: Path to the target image where we want to detect objects.
        ref_box: Bounding box in the reference image [x0, y0, x1, y1] (absolute coordinates).
        checkpoint: Path to SAM3 checkpoint.
        model_type: SAM3 model type.
        device: 'cuda' or 'cpu'.
    """
    
    print(f"Loading model {model_type} from {checkpoint}...")
    sam3 = sam3_model_registry[model_type](checkpoint=checkpoint)
    sam3.to(device=device)
    sam3.eval()
    
    # We use the processor for preprocessing mainly
    # Note: confidence_threshold is used in our manual post-processing
    processor = Sam3Processor(sam3, device=device) 
    
    # Load images
    ref_image_pil = Image.open(ref_image_path).convert("RGB")
    target_image_pil = Image.open(target_image_path).convert("RGB")
    
    ref_w, ref_h = ref_image_pil.size
    target_w, target_h = target_image_pil.size
    
    # Preprocess image batch
    print("Processing images...")
    # The processor.set_image_batch returns state with backbone features
    # But it assumes we are using the processor's state management. 
    # We will invoke set_image_batch to get the populated state.
    state = processor.set_image_batch([ref_image_pil, target_image_pil])
    
    # Prepare Prompt (Exemplar)
    # Convert absolute [x0, y0, x1, y1] to normalized [cx, cy, w, h]
    x0, y0, x1, y1 = ref_box
    box_w = x1 - x0
    box_h = y1 - y0
    cx = (x0 + x1) / 2 / ref_w
    cy = (y0 + y1) / 2 / ref_h
    nw = box_w / ref_w
    nh = box_h / ref_h
    
    norm_box = [cx, cy, nw, nh]
    
    print(f"Reference Box (Norm [cx, cy, w, h]): {norm_box}")
    
    # Construct FindStage manually for cross-image prompting
    # img_ids=[0] means the prompt comes from the first image in the batch (Reference Image)
    find_inputs = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long), # Dummy
        
        # input_boxes shape: [Batch=1, Seq=1, 4]
        input_boxes=torch.tensor([norm_box], device=device, dtype=torch.float32).view(1, 1, 4),
        input_boxes_mask=torch.tensor([[True]], device=device, dtype=torch.bool),
        input_boxes_label=torch.tensor([[True]], device=device, dtype=torch.long), # Positive prompt
        
        input_points=None,
        input_points_mask=None,
    )
    
    # We also need a dummy geometric prompt for the targets (required by forward_grounding usually)
    geometric_prompt = sam3._get_dummy_prompt()
    
    print("Running inference...")
    with torch.inference_mode():
        # Forward pass
        # backbone_out contains features for both images (Batch=2)
        outputs = sam3.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_inputs,
            geometric_prompt=geometric_prompt,
            find_target=None, # Search in all images (or logic handles it)
        )
        
        # Process outputs for Target Image (Index 1)
        # outputs["pred_boxes"] shape: [Batch, Queries, 4]
        # We want predictions for the second image (Index 1)
        
        target_idx = 1
        
        out_bbox = outputs["pred_boxes"][target_idx] 
        out_logits = outputs["pred_logits"][target_idx]
        # out_masks = outputs["pred_masks"][target_idx] # If needed
        
        # Scores
        out_probs = out_logits.sigmoid()
        # Presence score (objectness)
        if "presence_logit_dec" in outputs:
             # shape [Batch, Queries] or [Batch, 1]? 
             # Usually [Batch, 1] or similar. Let's inspect or assume per-batch.
             # Based on Sam3Processor._forward_grounding:
             # presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
             # out_probs = (out_probs * presence_score).squeeze(-1)
             
             presence_score = outputs["presence_logit_dec"][target_idx].sigmoid()
             # Multiply probabilities
             out_probs = out_probs * presence_score
        
        out_probs = out_probs.squeeze(-1) # Ensure flat scores if needed
        
        # Filter by threshold
        keep = out_probs > processor.confidence_threshold
        
        final_probs = out_probs[keep]
        final_boxes_norm = out_bbox[keep]
        
        # Convert boxes back to absolute [x0, y0, x1, y1] for target image
        # Pred boxes are [cx, cy, w, h] normalized? 
        # Sam3Processor uses box_ops.box_cxcywh_to_xyxy(out_bbox)
        
        final_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(final_boxes_norm)
        
        # Scale to target image dimensions
        scale_fct = torch.tensor([target_w, target_h, target_w, target_h], device=device)
        final_boxes_abs = final_boxes_xyxy * scale_fct[None, :]
        
        # Setup results
        final_boxes_np = final_boxes_abs.cpu().numpy()
        final_scores_np = final_probs.cpu().numpy()
        
        print(f"Found {len(final_boxes_np)} objects in target image.")
        
        show_results(target_image_pil, final_boxes_np, final_scores_np, "sam3_result.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 Cross-Image Detection")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to reference image")
    parser.add_argument("--target_image", type=str, required=True, help="Path to target image")
    parser.add_argument("--box", type=int, nargs=4, required=True, help="Ref Box x0 y0 x1 y1")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="Path to SAM3 checkpoint")
    parser.add_argument("--model", type=str, default="vit_l", help="Model type")
    
    args = parser.parse_args()
    
    run_cross_image_detection(
        args.ref_image,
        args.target_image,
        args.box,
        args.checkpoint,
        model_type=args.model
    )
