"""torchvision.ops.deform_conv2d rebuilt from grid_sample so Core ML can convert it."""
import torch, torch.nn.functional as F

def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)

def deform_conv2d_gs(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    B, C, H, W = input.shape
    Cout, Cin, kh, kw = weight.shape
    sh, sw = _pair(stride); ph, pw = _pair(padding); dh, dw = _pair(dilation)
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    ys = (torch.arange(Ho, dtype=input.dtype, device=input.device) * sh - ph).view(1, Ho, 1)
    xs = (torch.arange(Wo, dtype=input.dtype, device=input.device) * sw - pw).view(1, 1, Wo)
    cols = []
    K = kh * kw
    for i in range(kh):
        for j in range(kw):
            k = i * kw + j
            py = ys + i * dh + offset[:, 2 * k]
            px = xs + j * dw + offset[:, 2 * k + 1]
            gx = 2 * px / (W - 1) - 1
            gy = 2 * py / (H - 1) - 1
            grid = torch.stack([gx, gy], dim=-1)                       # [B,Ho,Wo,2]
            s = F.grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            if mask is not None:
                s = s * mask[:, k:k + 1]
            cols.append(s)
    cols = torch.cat(cols, dim=1)                                        # [B, K*C, Ho, Wo], index k*C + c
    w = weight.permute(0, 2, 3, 1).reshape(Cout, K * Cin, 1, 1)
    return F.conv2d(cols, w, bias)

if __name__ == "__main__":
    from torchvision.ops import deform_conv2d
    torch.manual_seed(0)
    for ks, pad in [(3, 1), (1, 0), (3, 2)]:
        x = torch.randn(1, 8, 20, 24); w = torch.randn(6, 8, ks, ks); b = torch.randn(6)
        off = torch.randn(1, 2 * ks * ks, 20 + 2 * pad - ks + 1, 24 + 2 * pad - ks + 1) * 2
        m = torch.rand(1, ks * ks, off.shape[2], off.shape[3])
        ref = deform_conv2d(x, off, w, b, padding=pad, mask=m)
        out = deform_conv2d_gs(x, off, w, b, padding=pad, mask=m)
        print(ks, pad, "max abs diff", (ref - out).abs().max().item())
