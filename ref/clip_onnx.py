import torch
import open_clip

# --- 1. Load CLIP model ---
# Load on CPU so it works even without GPU
# 'dfn5b' is the pretrained weight for ViT-H-14-378-quickgelu
print("Loading open_clip model...")
model, _, _ = open_clip.create_model_and_transforms(
    'ViT-H-14-378-quickgelu',
    pretrained='dfn5b',
    device=torch.device('cpu')
)
model.eval()

# --- 2. Image Encoder (Vision model) export with FIXED batch size = 1 ---
vision_model = model.visual

# Dummy input with fixed batch size = 1
dummy_image = torch.randn(1, 3, 378, 378)

print("Exporting Image Encoder (ViT-H-14-378) with fixed batch size 1 ...")
torch.onnx.export(
    vision_model,
    dummy_image,
    "clip_image_encoder.onnx",
    input_names=['image_input'],
    output_names=['image_features'],
    opset_version=20,
    # ❌ no dynamic_axes -> batch dimension becomes fixed as 1 in the ONNX graph
)
print("Image Encoder ONNX export complete.")


# --- 3. Text Encoder wrapper (to match CLIP text path) ---
class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final

        # keep attention mask as buffer so it's serialized in the model
        self.register_buffer('attn_mask', clip_model.attn_mask)

    def forward(self, text_tokens):
        # mirrors open_clip's encode_text logic up to pre-projection features
        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        return x

text_model = TextEncoderWrapper(model)
text_model.eval()

# CLIP standard context length is 77 tokens
dummy_text = torch.randint(0, 49408, (1, 77))  # (batch=1, seq_len=77)

print("\nExporting Text Encoder with dynamic batch...")
torch.onnx.export(
    text_model,
    dummy_text,
    "clip_text_encoder.onnx",
    input_names=['text_input_ids'],
    output_names=['text_features'],
    opset_version=20,
    dynamic_axes={
        'text_input_ids': {0: 'batch_size'}  # keep text encoder batch dynamic
    }
)
print("Text Encoder ONNX export complete.")

