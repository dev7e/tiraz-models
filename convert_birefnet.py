"""BiRefNet-lite (MIT) → Core ML mlprogram for Tiraz.

Input:  RGB image 1024×1024 (ImageType, 0…255 → scaled to 0…1 inside; ImageNet
        normalisation lives in the wrapper so the app just hands over pixels).
Output: "matte" grayscale image 1024×1024, 0…255 (sigmoid × 255).
"""
import sys, torch, coremltools as ct
from transformers import AutoModelForImageSegmentation

SIZE = 1024
out_path = sys.argv[1] if len(sys.argv) > 1 else "BiRefNetLite.mlpackage"

model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True)
model.eval()

# Core ML has no deformable convolution: swap torchvision's op for the
# grid_sample rebuild (bit-identical up to float error, see deform_gs.py).
from deform_gs import deform_conv2d_gs
mod = sys.modules[type(model).__module__]
mod.deform_conv2d = deform_conv2d_gs

class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):            # x: [1,3,H,W] in 0…1
        x = (x - self.mean) / self.std
        out = self.m(x)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        return torch.sigmoid(pred) * 255.0   # [1,1,H,W] 0…255

w = Wrapper(model).eval()
example = torch.rand(1, 3, SIZE, SIZE)
with torch.no_grad():
    traced = torch.jit.trace(w, example, strict=False)

ml = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
    outputs=[ct.ImageType(name="matte", color_layout=ct.colorlayout.GRAYSCALE)],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
    compute_units=ct.ComputeUnit.ALL,
)
ml.short_description = "BiRefNet-lite subject matting (MIT) — Tiraz high-quality background removal"
ml.author = "ZhengPeng7/BiRefNet (MIT); packaged for Tiraz"
ml.save(out_path)
print("saved", out_path)
