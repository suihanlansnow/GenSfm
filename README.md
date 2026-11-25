<div align="center">
<h1>GenSfM</h1>
<h3>Generative Structure from Motion with Native 3D Diffusion</h3>
<p><strong>NeurIPS 2026</strong></p>
<p>Haobin Duan<sup>1,2,&#42;</sup> · Binbin Huang<sup>1,2,&#42;</sup> · Mochu Xiang<sup>1,2</sup> · Yiqun Zhao<sup>1,2</sup><br>Zibo Zhao<sup>3</sup> · Yi Ma<sup>1,2,4</sup> · Shenghua Gao<sup>1,2,4,†</sup></p>
<p><sup>1</sup> The University of Hong Kong &nbsp; <sup>2</sup> Transengram &nbsp; <sup>3</sup> Tencent<br><sup>4</sup> Shenzhen Loop Area Institute<br><sup>&#42;</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding author</p>
<p><a href="https://arxiv.org/abs/2610.35522"><img src="assets/badges/paper.svg" alt="Paper"></a> <a href="https://dotb1ue.github.io/GenSfm/"><img src="assets/badges/project.svg" alt="Project page"></a> <a href="https://huggingface.co/dotblue/GenSfm"><img src="assets/badges/models.svg" alt="Models"></a> <a href="https://dotb1ue.github.io/GenSfm/#video"><img src="assets/badges/video.svg" alt="Video"></a></p>
<p><strong>A few images. One complete object. Its cameras.</strong></p>
<a href="https://dotb1ue.github.io/GenSfm/#gallery"><img src="assets/overview.webp" width="960" alt="Four views of a fabric wagon, its textured 3D reconstruction inside a dashed unit cube, and recovered cameras with fading denoising trajectories"></a>
</div>

GenSfM jointly recovers **3D geometry, appearance, and camera poses** from a few uncalibrated images of one object. A native 3D diffusion model generates shared shape and per-view pose latents; a pose-aware second stage recovers detailed geometry and texture using the estimated cameras.

[Quick start](#quick-start) · [Python API](#python-api) · [Outputs](#outputs) · [Training and evaluation](#training-and-evaluation)

> **Fork note / 分支说明** — This is a personal engineering fork of GenSfM. The model and API are unchanged; the inference path is optimized to run on a single consumer GPU under Windows. Why and how: [English](#purpose-and-engineering-optimizations) · [简体中文](#本仓库目的与工程优化).

## Quick start

### 1. Install

Use **Linux, an NVIDIA GPU, and the CUDA toolkit** (`nvcc`). Initialize Miniforge/Conda and Mamba in your shell. The default environment uses **Python 3.12, PyTorch 2.9.1, and CUDA 12.8**; use a compatible system CUDA toolkit to build the rendering extensions.

```bash
git clone --recurse-submodules https://github.com/dotb1ue/GenSfm.git
cd GenSfm

. ./setup.sh --new-env --basic --spconv --mipgaussian --nvdiffrast
mamba activate gensfm

export ATTN_BACKEND=sdpa
export ATTN_BACKEND_WITH_ATTN_BIAS=sdpa
export SPCONV_ALGO=native
```

`--new-env` creates the `gensfm` environment. Omit it to use an existing environment with compatible PyTorch/CUDA. PyTorch SDPA needs no optional attention packages. For an existing clone, run `git submodule update --init --recursive`. See `. ./setup.sh --help` for installation options.

### 2. Run the included example

```bash
python example_multi_image.py \
  --input-dir assets/example/wagon \
  --pretrained dotblue/GenSfm \
  --output-dir outputs/wagon
```

The first run downloads the [pretrained models](https://huggingface.co/dotblue/GenSfm) (approximately 5.8 GB) and the DINOv2 image encoder through Torch Hub. Open `outputs/wagon/mesh.glb` in a 3D viewer and compare `reprojection-00.png` through `reprojection-03.png` with the saved inputs. Choose a new output directory for each run.

Five example sets are included, each with four masked views:

| Input directory | Object | Images |
| --- | --- | --- |
| `assets/example/spongebob` | SpongeBob plush | Photographs |
| `assets/example/head-sculpture` | Metal head sculpture | Photographs from NeRD |
| `assets/example/garden-gnome` | Garden gnome | Photographs from Stanford-ORB |
| `assets/example/floral-teapot` | Floral teapot | Photographs from Stanford-ORB |
| `assets/example/wagon` | Fabric wagon | Synthetic ABO renders |

To try another set, replace `--input-dir` and choose a new `--output-dir` in the command above. The images retain their original source terms; see [asset attribution](assets/ATTRIBUTION.md).

### 3. Use your own images

Prepare **two to four views of the same rigid object**; four views covering different sides are recommended. Supply foreground-masked RGBA images with transparent backgrounds, keeping the complete object, holes, and thin parts in the mask.

```bash
python example_multi_image.py \
  --images /path/to/object/00.png /path/to/object/01.png \
           /path/to/object/02.png /path/to/object/03.png \
  --pretrained dotblue/GenSfm \
  --output-dir outputs/my-object
```

For RGB photographs, add `--remove-background` to create masks with rembg; inspect the masks when using cluttered backgrounds. Existing foreground alpha is preserved.

`--images` preserves the supplied order. `--input-dir` loads all supported images in filename order, so keep one object's views in that directory. The seed defaults to `1`; `--steps` sets the denoising steps in both stages. `--pretrained` also accepts a downloaded bundle directory; `--revision` selects a Hugging Face revision.

## Python API

Set the attention environment variables shown above before importing the pipeline.

```python
from trellis.pipelines import GenSfMPipeline

pipeline = GenSfMPipeline.from_pretrained("dotblue/GenSfm").cuda()
result = pipeline.run(
    [f"assets/example/wagon/{i:02d}.png" for i in range(4)],
    seed=1,
)

mesh = result["mesh"][0]
gaussians = result["gaussian"][0]
extrinsics, intrinsics = result["extrinsics"], result["intrinsics"]
```

The pipeline returns a mesh, Gaussians, and predicted cameras. Use the CLI to save a textured GLB, PLY, and reprojection images.

## Outputs

| File | Contents |
| --- | --- |
| `mesh.glb` | Textured mesh, in Y-up coordinates |
| `gaussian.ply` | 3D Gaussians |
| `cameras.json` | Predicted cameras, source-image intrinsics, and the data-to-mesh transform |
| `input-XX.png` / `input-rgba-XX.png` | Exact model inputs / cropped RGBA inputs |
| `reprojection-XX.png` | Textured-mesh renders at the predicted cameras, on black |
| `run.json` | Input order, crop mapping, seed, and inference settings |

Camera extrinsics are **OpenCV world-to-camera** matrices in the model's Z-up frame. Intrinsics are normalized matrices, including skew and principal-point offsets, for both the model crop and the EXIF-oriented source image. The GLB uses `(x, z, -y)` instead of model coordinates `(x, y, z)`; apply the `data_to_mesh` transform from `cameras.json` when projecting the exported mesh.

## Training and evaluation

Follow [DATASET.md](DATASET.md) to download data, render views, voxelize shapes, and encode training targets. UV target encoding requires a trained UV encoder, which is not part of the inference bundle.

| Component | Configuration |
| --- | --- |
| UV pose autoencoder | [uv_vae_conv3d_4l8_fp16.json](configs/vae/uv_vae_conv3d_4l8_fp16.json) |
| Joint shape and camera generation | [suv_flow_imgmv4vis_dit_L_16l8_p4l8_fp16.json](configs/generation/suv_flow_imgmv4vis_dit_L_16l8_p4l8_fp16.json) |
| Pose-aware geometry and appearance | [slat_flow_imgmv4_dit_L_64l8p2_fp16-convnextvisuallatentcond.json](configs/generation/slat_flow_imgmv4_dit_L_64l8p2_fp16-convnextvisuallatentcond.json) |

Install the training extras and download the two upstream denoisers used to initialize generator training:

```bash
. ./setup.sh --train --diffoctreerast

hf download microsoft/TRELLIS-image-large \
  --include 'ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors' \
            'ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors' \
  --local-dir checkpoint/TRELLIS-image-large
```

Match the configuration's latent directory names and UV decoder to your prepared data, then run stage 1:

```bash
python train.py \
  --config configs/generation/suv_flow_imgmv4vis_dit_L_16l8_p4l8_fp16.json \
  --data_dir /path/to/prepared/ABO,/path/to/prepared/HSSD \
  --output_dir outputs/gensfm-stage1 \
  --num_gpus 1 --auto_retry 0
```

Use the stage-2 configuration and a new output directory to train the appearance model. The two generators train independently. Adjust `trainer.args.batch_size_per_gpu` and its gradient-accumulation split to fit your hardware; for example, add `--override trainer.args.batch_size_per_gpu=1 trainer.args.batch_split=1`. Run the same command again to resume from the latest complete checkpoint. `python train.py --help` lists resume and distributed options.

For quantitative evaluation, install a [PyTorch3D build](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) compatible with your PyTorch/CUDA environment, then run `. ./setup.sh --eval`. `python eval.py --help` describes dataset evaluation. A complete two-stage evaluation must use the chosen appearance model's normalization and sampler; `sample_ssuv=false` evaluates with prepared conditioning geometry/cameras rather than the complete image-only pipeline.

<details>
<summary>Using your own training checkpoints</summary>

Keep each denoiser inside its training run's `ckpts/` directory, with the matching `config.json` one level above. Update `eval_models` dependencies to the correct trained UV decoder and appearance decoders. Named model configurations load local checkpoint files; model prefixes load matching JSON/safetensors files locally or from Hugging Face.

```bash
python example_multi_image.py \
  --input-dir /path/to/object \
  --stage1 /path/to/stage1/ckpts/denoiser.pt \
  --stage2 /path/to/stage2/ckpts/denoiser.pt \
  --output-dir outputs/my-checkpoints
```

Use `--pretrained` or both local checkpoint arguments. Full training continuation requires the original training state.

</details>

## Purpose and engineering optimizations

This fork targets a **single consumer GPU (RTX 3070, 8 GB) on Windows**. Upstream GenSfM is meant for multi-GPU Linux servers. The question: how much inference latency can be removed without changing outputs?

Three rules:

1. **Equal or better output.** Faster paths must be mathematically equivalent or quality-neutral on held-out views.
2. **Profile first.** Only stages that dominate wall-clock time are worth touching.
3. **Keep the interface.** Entry points and file formats stay compatible with upstream.

### Optimizations applied

| Stage | What we did | Speedup | How we verified |
| --- | --- | --- | --- |
| SLat → mesh (FlexiCubes 257³) | crop to active bounding box + 1-D integer key dedup (`torch.unique(dim=0)`) | **23×** | vertex count, face indices identical; `max\|Δv\| = 0` |
| Texture baking | closed-form normal-equation solver with Jacobi-CG + Huber-IRLS (replaces 2500 Adam steps), pass2 warm start, 8-round neighbourhood fill | **4.2×**, quality ↑ | PSNR on held-out views ≥ baseline |
| Per-seed camera solve (4× DLT SVD) | per-view subsample to ~8k valid voxels + batch all 4 views into one SVD call | **7×** | silhouette IoU unchanged across seeds |
| Bake pass 2 CG | warm start from pass 1 + every-5-step tol check + cap iters at 40 | incremental | pass residual < 1e-4, no quality change |
| Hole fill (8 rounds vs 32) | `solve()` already spreads values via TV regularization; residual propagation has diminishing returns after 8 rounds | incremental | no visible speckles, same postprocess mesh |

Total: **224 s → 120 s** for a 4-view, 3-seed run (excluding model loading).

![Performance comparison](assets/optimization/performance-comparison.png)

Cropped extraction and key-based dedup are on by default. `to_glb` keeps upstream's `bake_texture(mode='opt')`; the closed-form path is opt-in and typically faster.

## 本仓库目的与工程优化

本分支面向**单张消费级显卡（RTX 3070，8 GB）+ Windows** 的使用场景，上游 GenSfM 面向多卡 Linux 服务器。核心问题：在不改变任何输出的前提下，还能去掉多少推理耗时？

三条原则：

1. **输出不降级。** 更快的路径要么数学等价，要么在留出视角上持平或更好。
2. **先量化再优化。** 只动真正占大头的阶段。
3. **接口不变。** 入口、参数、文件格式与上游兼容。

### 优化项汇总

| 阶段 | 做法 | 加速 | 验收方式 |
| --- | --- | --- | --- |
| SLat → 网格（FlexiCubes 257³） | 裁剪到活跃包围盒 + 1-D 整数键去重 | **23×** | 顶点数、面索引完全一致；`max\|Δv\| = 0` |
| 贴图烘焙 | 闭式法方程求解 + Jacobi-CG + Huber-IRLS（替代 2500 Adam 步）+ CG warm start + 8 轮补洞 | **4.2×**，质量不降反升 | 留出视角 PSNR ≥ 基线 |
| 每 seed 相机求解（4× DLT SVD） | 每视角下采样到 ~8k 有效体素 + 4 视角 batched SVD | **7×** | 各 seed 轮廓 IoU 无变化 |
| 烘焙 pass 2 CG | 用 pass 1 解作初值 + 每 5 步 tol 检查 + iters 上限 40 | 小幅 | pass 残差 < 1e-4，质量无变化 |
| 补洞 8 轮（替代 32 轮） | `solve()` 已通过 TV 正则扩散了一层值，8 轮后蔓延递减 | 小幅 | 无可见麻点，后处理网格一致 |

4 视角、3 seed 场景，不含模型加载：**224 s → 120 s**。

![性能对比](assets/optimization/performance-comparison.png)

裁剪提取与键去重已默认开启。`to_glb` 仍保留上游的 `bake_texture(mode='opt')`；闭式烘焙需显式调用，通常更快。

## Citation

```bibtex
@inproceedings{duan2026gensfm,
  title = {Generative Structure from Motion with Native {3D} Diffusion},
  author = {Duan, Haobin and Huang, Binbin and Xiang, Mochu and Zhao, Yiqun
            and Zhao, Zibo and Ma, Yi and Gao, Shenghua},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  eprint = {2610.35522},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2610.35522}
}
```

## Acknowledgements and license

GenSfM builds on [TRELLIS](https://github.com/microsoft/TRELLIS), including its Structured LATent representation, decoders, and data tooling. We also thank [DINOv2](https://github.com/facebookresearch/dinov2) and the rendering libraries used by this project.

Our code uses the [MIT License](LICENSE). Third-party licenses and original notices, including TRELLIS, are in [THIRD_PARTY.md](THIRD_PARTY.md). See [asset attribution](assets/ATTRIBUTION.md) for example image sources and terms.
