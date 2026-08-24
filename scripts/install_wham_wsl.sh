#!/usr/bin/env bash
set -euo pipefail

install_root="${POSE_WHAM_INSTALL_ROOT:-$HOME/.local/share/pose-wham}"
conda_root="$install_root/miniconda"
wham_root="$install_root/WHAM"
mkdir -p "$install_root"

if [[ ! -x "$conda_root/bin/conda" ]]; then
  installer="$install_root/miniconda.sh"
  curl -fL --retry 3 \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o "$installer"
  bash "$installer" -b -p "$conda_root"
  rm -f "$installer"
fi

source "$conda_root/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx wham; then
  conda create -y -n wham python=3.9 --override-channels -c conda-forge
fi
conda activate wham
python -m pip install --upgrade "pip<25" wheel

if [[ ! -d "$wham_root/.git" ]]; then
  for attempt in 1 2 3; do
    if git -c http.version=HTTP/1.1 clone \
      https://github.com/yohanshin/WHAM.git "$wham_root"; then
      break
    fi
    if [[ "$attempt" == 3 ]]; then
      echo "Unable to clone WHAM after three attempts" >&2
      exit 1
    fi
    sleep 3
  done
fi
for attempt in 1 2 3; do
  if git -C "$wham_root" -c http.version=HTTP/1.1 \
    submodule update --init --depth 1 --force third-party/ViTPose; then
    break
  fi
  if [[ "$attempt" == 3 ]]; then
    echo "Unable to clone ViTPose after three attempts" >&2
    exit 1
  fi
  sleep 3
done

# CUDA 11.8 supports the RTX 40-series and remains close to WHAM's original
# dependency generation. DPVO is intentionally omitted: HYROX assistance uses
# local 3D joints and does not require camera SLAM.
python -m pip install \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install \
  "numpy==1.23.5" "setuptools==59.5.0" \
  yacs joblib scikit-image opencv-python 'imageio[ffmpeg]' matplotlib \
  tensorboard smplx progress einops "mmcv==1.3.9" "timm==0.4.9" \
  munkres "xtcocotools>=1.8" loguru tqdm ultralytics "gdown==4.6.0" \
  "chumpy @ git+https://github.com/mattloper/chumpy"
python -m pip install -v -e "$wham_root/third-party/ViTPose"

mkdir -p "$wham_root/dataset/body_models" "$wham_root/checkpoints"

download_gdrive() {
  local file_id="$1"
  local target="$2"
  if [[ ! -s "$target" ]]; then
    python -m gdown "https://drive.google.com/uc?id=${file_id}&export=download&confirm=t" -O "$target"
  fi
}

body_archive="$install_root/body_models.tar.gz"
download_gdrive 1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ "$body_archive"
if [[ ! -f "$wham_root/dataset/body_models/J_regressor_wham.npy" ]]; then
  tar -xzf "$body_archive" -C "$wham_root/dataset"
fi

download_gdrive 1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB \
  "$wham_root/checkpoints/wham_vit_w_3dpw.pth.tar"
download_gdrive 19qkI-a6xuwob9_RFNSPWf1yWErwVVlks \
  "$wham_root/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar"
download_gdrive 1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y \
  "$wham_root/checkpoints/hmr2a.ckpt"
download_gdrive 1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo \
  "$wham_root/checkpoints/yolov8x.pt"
download_gdrive 1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso \
  "$wham_root/checkpoints/vitpose-h-multi-coco.pth"

smpl_model="$wham_root/dataset/body_models/smpl/SMPL_NEUTRAL.pkl"
mkdir -p "$(dirname "$smpl_model")"
if [[ ! -f "$smpl_model" ]]; then
  cat >&2 <<EOF
WHAM_PUBLIC_COMPONENTS_INSTALLED
The licensed SMPL neutral model is still required at:
  $smpl_model
Register and accept the model license at https://smplify.is.tue.mpg.de/,
then copy basicModel_neutral_lbs_10_207_0_v1.0.0.pkl to that path as
SMPL_NEUTRAL.pkl. The project must not redistribute this licensed file.
EOF
  exit 2
fi

cd "$wham_root"
python - <<'PY'
import torch
from configs.config import get_cfg_defaults
from lib.models import build_body_model

assert torch.cuda.is_available(), "CUDA is not available inside WSL"
model = build_body_model("cuda", 1)
print("WHAM_PREFLIGHT_OK", torch.cuda.get_device_name(), type(model).__name__)
PY

echo "WHAM_INSTALLATION_COMPLETE $wham_root"
