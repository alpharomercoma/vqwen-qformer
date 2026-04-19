"""Architectural constants.

MiniGPT-4 recipe:
  EVA-ViT-G/14 (224x224)  [frozen, from BLIP-2]
    -> 257 tokens x 1408 hidden
  Q-Former (12 layers)    [frozen, from BLIP-2]
    -> 32 query tokens x 768 hidden (queries are pretrained, not random)
  Linear 768 -> 2560      [TRAINABLE]
  Qwen3-4B                [frozen stage-1; LoRA-adapted stage-2]
"""

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
NUM_QUERY_TOKENS = 32
QFORMER_HIDDEN_SIZE = 768
