# tiraz-models

Core ML models the Tiraz iOS app downloads on demand (they are not bundled in the App Store binary).

## BiRefNet-lite — subject matting

Release tag `birefnet-lite-v1`. The compiled `.mlmodelc` is published as individual files plus
`manifest.json` (paths, sizes, SHA-256). The app fetches the manifest, downloads each file in the
background, verifies it, rebuilds the `BiRefNetLite.mlmodelc` folder and loads it with `MLModel(contentsOf:)`.

- Input `image`: RGB 1024×1024. Output `matte`: grayscale 1024×1024 (0 = background, 255 = subject).
- ImageNet normalisation is inside the model; hand it plain pixels.
- Converted from [ZhengPeng7/BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) with
  coremltools 9 (fp16, iOS 17+) — see `convert_birefnet.py`. The deformable convolutions are rebuilt
  from `grid_sample` for Core ML (`deform_gs.py`); numerically identical to torchvision's op.
- `publish-release.sh` builds the manifest and uploads a compiled model as a release.

Model weights and architecture: © Peng Zheng et al., MIT License (see LICENSE).
