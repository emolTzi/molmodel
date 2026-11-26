# 单物体去背景（BiRefNet 首帧 + SAM 2.1 跟踪）

这个模式只减少背景在训练损失里的权重，不会补回主体没有拍到的细节，也不会改原始素材或把背景涂黑后冒充遮罩。

默认关闭。旧项目没有该字段时仍按整场景处理。照片、多组素材、透明视频和全景视频不能使用，界面会说明原因。

开启后有两条路，都还是交给 SAM 2.1 做整段跟踪：

- 自动首帧：用官方 [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)（MIT）给第 0 帧出一张掩码，再交给 SAM 2.1 的 `add_new_mask` 往后跟踪。不需要点选。
- 手动修正：原来的前景点、背景点和主体框仍然可用。它走 SAM 2.1 的点/框提示，不经过 BiRefNet。

不会改用 rembg 的默认 Bria RMBG 权重，也不会用 AGPL 的 background-removal-js。路径里如果出现 Bria、rembg 这类名字，预检和分割都会停住。

## 本机准备

不要由 OOOSplat 下载权重。代码与检查点的许可证是 [Apache-2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE)。下面的字节数是 2026-11-19 对官方文件做 HTTP `Content-Length` 得到的，不是估计。

官方 README 表格里的 “Size (M)” 是参数量，不是文件大小。

| 文件 | 配置名 | 字节 | 来源 |
| --- | --- | ---: | --- |
| `sam2.1_hiera_tiny.pt` | `configs/sam2.1/sam2.1_hiera_t.yaml` | 156008466 | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt |
| `sam2.1_hiera_small.pt` | `configs/sam2.1/sam2.1_hiera_s.yaml` | 184416285 | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt |
| `sam2.1_hiera_base_plus.pt` | `configs/sam2.1/sam2.1_hiera_b+.yaml` | 323606802 | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt |
| `sam2.1_hiera_large.pt` | `configs/sam2.1/sam2.1_hiera_l.yaml` | 898083611 | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt |

建议先用 tiny。安装官方仓库（Windows 上官方建议用 WSL Ubuntu；原生 Windows 若能装上 CUDA 版 PyTorch 也可以，预检不过就不要开始分割）：

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

另外需要 Python 3.10 或更高、`torch>=2.5.1`、`torchvision>=0.20.1`，以及带 CUDA 的本机 GPU。安装说明见 https://github.com/facebookresearch/sam2/blob/main/README.md 和 https://pytorch.org/get-started/locally/ 。

在界面里填写这个 Python 可执行文件和已经下载好的 `.pt` 路径。预检失败时，分割和“开始生成”都会停住。

## 自动首帧（BiRefNet）

SAM 2 仍然负责跟踪和手动修正。BiRefNet 只给第一帧做初始掩码。不要由 OOOSplat 下载权重，也不要只放一个 `.safetensors`。请自己下载完整的官方快照，目录里要有 `model.safetensors`、`config.json`、`birefnet.py`、`BiRefNet_config.py`。`config.json` 里的 `bb_pretrained` 必须是 `false`，避免再去下载骨干网络。

代码与权重的许可证是 [MIT](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE)。下面的字节数和 SHA256 是官方 Git LFS 指针里的值（Hugging Face 修订 `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`），不是在这台机器上重新下载后估出来的。

| 文件 | 字节 | SHA256 | 来源 |
| --- | ---: | --- | --- |
| `model.safetensors` | 444473596 | `9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154` | https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors |

另外需要能 `import transformers`。推理尺寸沿用官方说明里的 1024，分割进程会强制 Hugging Face 离线（`local_files_only`）。权重对不上、PyTorch 没有 CUDA，或者没有 transformers 时，自动首帧会失败并停住，不会假装已经分完。手动点选不依赖这份权重。

界面里把「BiRefNet 本地快照」指到这个目录，或指到其中的 `model.safetensors`，保存后再预检。

## 和训练的关系

COLMAP 仍使用带背景的原始抽帧。掩码只在 Brush 数据集的 `masks/` 目录里，并传给锁定的 OOOBrush `ooo-v1.0.0`（commit `866a7d65b4d6a591ba0d52a88f8b20ea3465c1e5`）的 `--alpha-mode masked`。该模式把掩码写进 alpha，损失乘以 alpha；白色保留，黑色忽略。它不是透明材质那种逼迫 splat 透明度贴合画面的模式。

如果正在使用的 `brush_app --help` 没有 `--alpha-mode` 和 `--invert-masks`，训练会在启动前停止。画面 RGB 不会被涂黑。

SAM 2 进程结束并释放显卡后才会进入重建和训练。
