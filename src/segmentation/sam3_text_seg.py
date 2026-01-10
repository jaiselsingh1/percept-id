import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor
import cv2

class Sam3TextSegmentor:
    """text conditioned segmentation using SAM-3"""
    def __init__(self, model_name: str = "sam3.pt", conf_threshold: float = 0.25):
        """Initialize SAM-3. Will auto-download model if not found."""
        self.model_name = model_name
        self.conf_threshold = conf_threshold

        print(f"Loading SAM-3 model: {model_name}...")

        # Initialize predictor with overrides
        overrides = dict(
            conf=conf_threshold,
            task="segment",
            mode="predict",
            model=model_name,
            half=True,
            save=False,
            verbose=False
        )

        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        print("SAM-3 loaded successfully")

    def segment_with_text(
            self,
            rgb: np.ndarray,
            text_prompts: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Segment image with text prompts.

        Args:
            rgb: RGB image as numpy array (H, W, 3) in range [0, 255] or [0, 1]
            text_prompts: List of text prompts to segment

        Returns:
            Dictionary mapping prompt -> list of binary masks
        """
        # Convert to uint8 if needed
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            if rgb.max() <= 1.5:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        # Convert RGB to BGR for OpenCV/Ultralytics
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Set the image once
        self.predictor.set_image(bgr)

        # Query all prompts at once
        print(f"Segmenting with prompts: {text_prompts}")
        masks, boxes = self.predictor(text=text_prompts)

        # Parse results - masks is list of lists, one per prompt
        results = {}
        for i, prompt in enumerate(text_prompts):
            prompt_masks = []
            if masks is not None and i < len(masks):
                # masks[i] contains all instances for this prompt
                if masks[i] is not None:
                    for j, mask in enumerate(masks[i]):
                        # Convert to boolean numpy array
                        if hasattr(mask, 'cpu'):
                            mask_np = mask.cpu().numpy().astype(bool)
                        else:
                            mask_np = np.array(mask).astype(bool)
                        prompt_masks.append(mask_np)
                        print(f"  '{prompt}' instance {j+1}: mask shape {mask_np.shape}")

            results[prompt] = prompt_masks
            print(f"Found {len(prompt_masks)} instances for '{prompt}'")

        return results
        

if __name__ == "__main__":
    from data_loaders.rgbd_loader import RGBDDataLoader
    import matplotlib.pyplot as plt

    print("Testing SAM-3 text segmentation...")

    loader = RGBDDataLoader(data_root="data")
    frame_data = loader.get_frame_data(0, camera_name='front')
    rgb = frame_data['cameras']['front']['rgb']

    # Initialize SAM-3 segmentor
    segmentor = Sam3TextSegmentor()

    # Segment with text prompts
    masks_dict = segmentor.segment_with_text(rgb, ["mug", "wooden dish rack"])

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Get first mug mask if any found
    if "mug" in masks_dict and len(masks_dict["mug"]) > 0:
        overlay = rgb.copy().astype(np.float32)
        mask = masks_dict["mug"][0]  # Use first mug
        overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title(f"Mug (RED) - {len(masks_dict['mug'])} found")
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No mug found', ha='center', va='center')
        axes[1].set_title("Mug - 0 found")
        axes[1].axis('off')

    # Get first rack mask if any found
    if "wooden dish rack" in masks_dict and len(masks_dict["wooden dish rack"]) > 0:
        overlay = rgb.copy().astype(np.float32)
        mask = masks_dict["wooden dish rack"][0]  # Use first rack
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title(f"Rack (GREEN) - {len(masks_dict['wooden dish rack'])} found")
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'No rack found', ha='center', va='center')
        axes[2].set_title("Rack - 0 found")
        axes[2].axis('off')

    plt.tight_layout()
    output_path = Path("outputs/visualizations/sam3_text_segmentation.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.show()




