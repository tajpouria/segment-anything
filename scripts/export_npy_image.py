import argparse
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

parser = argparse.ArgumentParser(description="Export NPY image embedding.")

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM model checkpoint.",
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

parser.add_argument("--image", type=str, required=True, help="The path to the image.")

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="The path to the image embedding output.",
)

checkpoint = parser.parse_args().checkpoint
model_type = parser.parse_args().model_type
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device="cuda")
predictor = SamPredictor(sam)

image = cv2.imread(parser.parse_args().image)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save(parser.parse_args().output, image_embedding)
