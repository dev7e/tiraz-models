#!/bin/zsh
# Publishes a compiled Core ML model to GitHub Releases on the PUBLIC repo
# dev7e/tiraz-models, as individual files + manifest.json (what the app's
# MatteModelStore downloads). Needs: `gh auth login` done, and the repo to
# exist (create it once: gh repo create dev7e/tiraz-models --public ...).
#
# Usage: publish-release.sh <compiled.mlmodelc dir> <tag>   e.g. birefnet-lite-v1
set -euo pipefail
DIR=${1:?compiled .mlmodelc directory}
TAG=${2:?release tag, e.g. birefnet-lite-v1}
REPO=dev7e/tiraz-models
WORK=$(mktemp -d)

cd "$DIR"
python3 - "$WORK/manifest.json" <<'EOF'
import hashlib, json, os, sys
files = []
for root, _, names in os.walk('.'):
    for n in names:
        p = os.path.join(root, n)[2:]
        files.append({"path": p, "asset": p.replace('/', '__'), "bytes": os.path.getsize(p),
                      "sha256": hashlib.sha256(open(p, 'rb').read()).hexdigest()})
files.sort(key=lambda f: f["path"])
json.dump({"model": "BiRefNetLite", "version": 1, "format": "mlmodelc", "inputSize": 1024,
           "minimumIOS": "17.0", "license": "MIT (ZhengPeng7/BiRefNet)", "files": files},
          open(sys.argv[1], 'w'), indent=2)
print("\n".join(f'{f["asset"]}  {f["bytes"]} bytes' for f in files))
EOF

# Asset names cannot contain "/" — flatten the path with "__" (the manifest maps them back).
for f in $(find . -type f | sed 's|^\./||'); do
  cp "$f" "$WORK/$(echo "$f" | sed 's|/|__|g')"
done

gh release create "$TAG" --repo "$REPO" \
  --title "BiRefNet-lite v1 (Core ML, fp16, iOS 17+)" \
  --notes "Compiled BiRefNetLite.mlmodelc as individual files + manifest.json (SHA-256). Input image 1024x1024 RGB, output matte 1024x1024 grayscale. MIT." \
  "$WORK"/*
echo "manifest: https://github.com/$REPO/releases/download/$TAG/manifest.json"
