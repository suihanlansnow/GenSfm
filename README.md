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

### Why this fork exists

Upstream GenSfM targets a multi-GPU Linux server. This fork targets a **single consumer GPU (RTX 3070, 8 GB) on Windows** and asks one question: how much inference latency can be removed without changing a single output?

Three rules guided the work:

1. **Equal or better output.** A faster path must be mathematically equivalent (verified element-wise) or quality-neutral/better on held-out views. Being faster alone is not a result.
2. **Profile first.** Only the stages that dominate wall-clock time are worth touching.
3. **Keep the interface.** Entry points, arguments, and file formats stay compatible with upstream.

### Where the time goes

Measured on a 4-view input with 3 seeds × 25 denoising steps, excluding model loading (≈224 s total):

| Stage | Time | Share |
| --- | --- | --- |
| SLat → mesh extraction (FlexiCubes, 257³) | 64.4 s | 29% |
| Texture baking (Adam, 2048², 2500 steps) | 43.4 s | 19% |

### Decision 1 — closed-form texture baking

The renderer is linear in the texel values, so baking colour onto a UV atlas is, per texel, an independent small weighted least-squares problem. Upstream solves it with 2500 Adam iterations — an iterative approximation of a problem that has a direct normal-equation solution.

`bake_texture_lsq` assembles the per-texel 3×3 stencils, solves `(WᵀW + λLap) x = Wᵀb` with Jacobi-preconditioned CG, then runs two Huber-IRLS rounds so specular highlights and mis-tracked pixels cannot dominate. Texels inside UV islands that no view observed are filled from neighbours; the remaining holes are inpainted.

Result: **43.4 s → 10.4 s**, and quality *improved* — a converged closed-form solution beats an early-stopped optimizer. Training-view PSNR 24.87 → 26.31 dB; held-out PSNR 23.23 → 23.24 dB (linear) and 23.58 → 23.59 dB (mipmap).

### Decision 2 — crop FlexiCubes to the active bounding box

Extraction ran on a dense 257³ grid although the latent occupies only a fraction of it. The key observation is an invariant: the SDF is written only at active-cube vertices and defaults to +1 elsewhere, so every surface cube contains at least one active vertex — cropping to `[lo-1, hi+1]` provably cannot drop surface.

The remaining cost was dominated by `torch.unique(..., dim=0)`. Replacing the 3-column lookup with a 1-D integer key preserves lexicographic order exactly while using a far cheaper sort; the same trick applies to surface-edge dedup inside FlexiCubes.

Result: **64.4 s → 2.8 s**, with identical vertex count, identical face indices, and `max|Δv| = 0`.

### Decision 3 — what was deliberately not shipped

Caching DINOv2 condition encodings across seeds saves ≈1.5 s (1.4%) and stayed out of the default path: real, but too small to justify extra state. The rule of thumb is that an optimization must be either free (provably equivalent) or large.

### Measured result

| Metric | Before | After | Change |
| --- | --- | --- | --- |
| End-to-end (without model loading) | 224.0 s | 130.4 s | 1.72× |
| Mesh extraction | 64.4 s | 2.8 s | 23× |
| Texture baking | 43.4 s | 10.4 s | 4.2× |
| PSNR, training views | 24.87 dB | 26.31 dB | +1.44 |
| PSNR, held-out (linear) | 23.23 dB | 23.24 dB | +0.01 |
| PSNR, held-out (mipmap) | 23.58 dB | 23.59 dB | +0.01 |

![Performance comparison](assets/optimization/performance-comparison.png)

Cropped extraction and key-based dedup are on by default. `to_glb` keeps the upstream `bake_texture(mode='opt')` choice; the baking numbers above were measured with `bake_texture_lsq` at 2048², 160 views and 768² rasterization, so that fast path is currently opt-in.

## 本仓库目的与工程优化

### 这个分支为什么存在

上游 GenSfM 面向多卡 Linux 服务器。本分支的目标是在**单张消费级显卡（RTX 3070，8 GB）+ Windows** 上跑通完整重建，并回答一个问题：在不改变任何输出的前提下，还能去掉多少推理耗时？

三条原则：

1. **输出不降级。** 更快的路径要么数学等价（逐元素校验），要么在留出视角上质量持平或更好。只是"更快"不算结果。
2. **先量化再优化。** 只动真正占大头的阶段，不做凭感觉的微优化。
3. **接口不变。** 入口、参数、文件格式与上游保持兼容。

### 瓶颈在哪里

4 视角输入、3 个 seed × 25 步去噪，除模型加载外约 224 s，其中：

| 阶段 | 耗时 | 占比 |
| --- | --- | --- |
| SLat → 网格提取（FlexiCubes，257³） | 64.4 s | 29% |
| 贴图烘焙（Adam，2048²，2500 步） | 43.4 s | 19% |

### 决策一：把贴图烘焙改成闭式求解

渲染对纹素取值是线性的，所以"把颜色烘到 UV 图集"本质上是每个纹素各自独立的小型加权最小二乘问题。上游用 2500 步 Adam 去逼近一个本来就有法方程闭式解的问题。

`bake_texture_lsq` 按纹素组装 3×3 stencil，用 Jacobi 预条件共轭梯度解 `(WᵀW + λLap) x = Wᵀb`，再做两轮 Huber-IRLS，避免高光与跟踪错误的像素主导结果；UV 岛内没有被任何视角覆盖的纹素先用邻域蔓延补洞，残余孔洞再 inpaint。

结果：**43.4 s → 10.4 s**，且质量不降反升——收敛的闭式解优于提前停止的迭代解：训练视角 PSNR 24.87 → 26.31 dB，留出视角 23.23 → 23.24 dB（线性）、23.58 → 23.59 dB（mipmap）。

### 决策二：把 FlexiCubes 裁到活跃包围盒

提取原本在 257³ 稠密网格上做，而隐式场只占其中一小块。关键是一个不变式：SDF 只在活跃立方体顶点赋值、其余默认为 +1，因此任何表面立方体必然包含至少一个活跃顶点——裁剪到 `[lo-1, hi+1]` 可证明不会丢表面。

其次，剩余耗时主要来自 `torch.unique(..., dim=0)`。把三列坐标换成 1-D 整数键，字典序完全一致而排序代价大幅下降；同样的办法也用在 FlexiCubes 内部的表面边去重上。

结果：**64.4 s → 2.8 s**，顶点数、面索引完全一致，`max|Δv| = 0`。

### 决策三：主动放弃的优化

跨 seed 缓存 DINOv2 条件编码只省约 1.5 s（1.4%），没有进入默认路径——收益真实但太小，不值得增加状态。判断标准很简单：优化要么"免费"（可证明等价），要么足够大。

### 结果

| 指标 | 优化前 | 优化后 | 变化 |
| --- | --- | --- | --- |
| 端到端（不含模型加载） | 224.0 s | 130.4 s | 1.72× |
| 网格提取 | 64.4 s | 2.8 s | 23× |
| 贴图烘焙 | 43.4 s | 10.4 s | 4.2× |
| PSNR，训练视角 | 24.87 dB | 26.31 dB | +1.44 |
| PSNR，留出视角（线性） | 23.23 dB | 23.24 dB | +0.01 |
| PSNR，留出视角（mipmap） | 23.58 dB | 23.59 dB | +0.01 |

![性能对比](assets/optimization/performance-comparison.png)

裁剪提取与键去重已默认开启。`to_glb` 仍保留上游的 `bake_texture(mode='opt')`；上表烘焙数据由 `bake_texture_lsq` 在 2048²、160 视角、768² 光栅化下测得，因此该快路径目前需显式调用。

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
