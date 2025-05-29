# Refer to https://huggingface.co/meimeilook/BAGEL-7B-MoT-FP8

vim run_edit.py
```python
from datasets import load_dataset
from gradio_client import Client, handle_file
from PIL import Image
from tqdm import tqdm
import os
import glob
from datasets import Image as HfImage

# 加载数据集
ds = load_dataset("svjack/HQ-Edit-Sample-2500")
output_dir = "output_images"  # 设置输出路径
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

# 初始化Gradio客户端
client = Client("http://127.0.0.1:7860/")

# 获取已经处理过的文件索引（防止重复处理）
processed_indices = set()
for f in glob.glob(os.path.join(output_dir, "*_output.png")):
    try:
        idx = int(os.path.basename(f).split("_")[0])
        processed_indices.add(idx)
    except:
        continue

print(f"Found {len(processed_indices)} already processed files")

# 处理所有样本（跳过已处理的）
for i in tqdm(range(len(ds["train"])), desc="Processing images"):
    if i in processed_indices:
        continue  # 跳过已处理的
    
    try:
        # 保存原始图像
        input_path = os.path.join(output_dir, f"{i:05d}_input.png")
        ds["train"][i]["start_image"].save(input_path)
        
        # 处理图像
        result = client.predict(
            image=handle_file(input_path),
            prompt=ds["train"][i]["edit_prompt"],
            show_thinking=False,
            cfg_text_scale=4,
            cfg_img_scale=2,
            cfg_interval=0,
            timestep_shift=3,
            num_timesteps=50,
            cfg_renorm_min=0,
            cfg_renorm_type="text_channel",
            max_think_token_n=1024,
            do_sample=False,
            text_temperature=0.3,
            seed=0,
            api_name="/process_edit_image"
        )
        
        # 保存处理后的图像
        output_path = os.path.join(output_dir, f"{i:05d}_output.png")
        Image.open(result[0]).save(output_path)
    except Exception as e:
        print(f"Error processing sample {i}: {str(e)}")
        continue

# 构建新数据集（通过读取output_images目录）
def create_new_dataset(original_ds, output_dir):
    # 获取所有处理后的文件（按数字顺序）
    output_files = sorted(
        glob.glob(os.path.join(output_dir, "*_output.png")),
        key=lambda x: int(os.path.basename(x).split("_")[0])
    )
    
    # 获取对应的原始样本索引
    indices = [int(os.path.basename(f).split("_")[0]) for f in output_files]
    
    # 选择对应的样本
    selected_samples = original_ds["train"].select(indices)
    
    # 添加处理后的图像列
    def add_processed_image(example, idx):
        example["BAGEL_Edit_image"] = os.path.join(output_dir, f"{idx:05d}_output.png")
        return example
    
    new_ds = selected_samples.map(add_processed_image, with_indices=True)
    
    # 转换图像列类型
    new_ds = new_ds.cast_column("BAGEL_Edit_image", HfImage())
    return new_ds

new_dataset = create_new_dataset(ds, output_dir)
print(f"Final dataset contains {len(new_dataset)} processed samples")
```

<p align="center">
  <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="480"/>
</p>

<p align="center">
  <a href="https://bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white"
      alt="BAGEL Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red"
      alt="BAGEL Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
    <img 
        src="https://img.shields.io/badge/BAGEL-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="BAGEL Model"
    />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"
      alt="BAGEL Demo"
    />
  </a>
  <a href="https://huggingface.co/spaces/ByteDance-Seed/BAGEL">
    <img 
        src="https://img.shields.io/badge/BAGEL-Space-orange?logo=huggingface&logoColor=yellow" 
        alt="BAGEL Model"
    />
  </a>
  <a href="https://discord.gg/Z836xxzy">
    <img
      src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple"
      alt="BAGEL Discord"
    />
  </a>
  <a href="mailto:bagel@bytedance.com">
    <img
      src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red"
      alt="BAGEL Email"
    />
  </a>
</p>

# Unified Model for Multimodal Understanding and Generation
> [Chaorui Deng*](https://scholar.google.com/citations?hl=en&user=k0TWfBoAAAAJ), [Deyao Zhu*](https://tsutikgiau.github.io/), [Kunchang Li*](https://andy1621.github.io/), [Chenhui Gou*](https://www.linkedin.com/in/chenhui-gou-9201081a1/?originalSubdomain=au), [Feng Li*](https://fengli-ust.github.io/), [Zeyu Wang](https://zw615.github.io/), Shu Zhong, [Weihao Yu](https://whyu.me/), [Xiaonan Nie](https://codecaution.github.io/), [Ziang Song](https://www.linkedin.com/in/ziang-song-43b0ab8a/), Guang Shi :email: , [Haoqi Fan* :tophat: ](https://haoqifan.github.io/)
>
> contact: shiguang.sg@bytedance.com
> 
> We present **BAGEL**, an open‑source multimodal foundation model with 7B active parameters (14B total) trained on large‑scale interleaved multimodal data. BAGEL outperforms the current top‑tier open‑source VLMs like Qwen2.5-VL and InternVL-2.5 on standard multimodal understanding leaderboards, and delivers text‑to‑image quality that is competitive with strong specialist generators such as SD3.
Moreover, BAGEL demonstrates superior qualitative results in classical image‑editing scenarios than the leading open-source models. More importantly, it extends to free-form visual manipulation, multiview synthesis, and world navigation, capabilities that constitute "world-modeling" tasks beyond the scope of previous image-editing models.
The figure below showcases BAGEL's qualitative performance.

<p align="center"><img src="assets/teaser.webp" width="95%"></p>


<!-- ## 🧠 Method
BAGEL adopts a Mixture-of-Transformer-Experts (MoT) architecture to maximize the model’s capacity to learn from richly diverse multimodal information. Following the same principle of capacity maximization, it utilizes two separate encoders to capture pixel-level and semantic-level features of an image. The overall framework follows a Next Group of Token Prediction paradigm, where the model is trained to predict the next group of language or visual tokens as a compression target.

BAGEL scales MoT’s capacity through Pre-training, Continued Training, and Supervised Finetuning on trillions of interleaved multimodal tokens spanning language, image, video, and web data. It surpasses open models on standard understanding and generation benchmarks and demonstrates advanced in-context multimodal abilities like free-form image editing, future frame prediction, 3D manipulation, world navigation, and sequential reasoning.

<p align="center"><img src="assets/arch.png" width="95%"></p>


## 🌱 Emerging Properties
<p align="center"><img src="assets/emerging_curves.png" width="95%"></p>

As we scale up BAGEL’s pretraining with more multimodal tokens, we observe consistent performance gains across understanding, generation, and editing tasks. Different capabilities emerge at distinct training stages—multimodal understanding and generation appear early, followed by basic editing, while complex, intelligent editing emerges later. This staged progression suggests an emergent pattern, where advanced multimodal reasoning builds on well-formed foundational skills. Ablation studies further show that combining VAE and ViT features significantly improves intelligent editing, underscoring the importance of visual-semantic context in enabling complex multimodal reasoning and further supporting its role in the emergence of advanced capabilities. -->

## 📢 News

We sincerely thank all contributors from the open community for their valuable support.

- **May 26, 2025:** Thanks to [@neverbiasu](https://github.com/neverbiasu) for contributing [ComfyUI](https://github.com/neverbiasu/ComfyUI-BAGEL).
- **May 25, 2025:** Special thanks to [@LeanModels](https://github.com/LeanModels) for providing the [DF11-compressed version](https://huggingface.co/DFloat11/BAGEL-7B-MoT-DF11), and to [@Gapeleon](https://huggingface.co/Gapeleon) for the [INT8-compressed version](https://huggingface.co/Gapeleon/bytedance_BAGEL-7B-MoT-INT8). We also appreciate [@gluttony-10](https://github.com/gluttony-10) for contributions to the [Windows package](https://github.com/ByteDance-Seed/Bagel/issues/51).
- **May 24, 2025:** Together with [@wangwei1237](https://github.com/wangwei1237), [@gluttony-10](https://github.com/gluttony-10), and [@KingNish24](https://github.com/KingNish24), we built a Gradio [app](app.py) and launched a [Hugging Face Space](https://huggingface.co/spaces/ByteDance-Seed/BAGEL).
- **May 23, 2025:** We have provided a training guideline in [TRAIN](./TRAIN.md).
- **May 20, 2025:** We released the official [website](https://bagel-ai.org/), [demo](https://demo.bagel-ai.org/), [model](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT), and [report](https://arxiv.org/abs/2505.14683) for BAGEL.


## 📮 Notice
**Call for Bad Cases:** If you have encountered any cases where the model performs poorly, we would greatly appreciate it if you could share them in the [issue#11](https://github.com/ByteDance-Seed/Bagel/issues/11) or [Discord](https://discord.gg/Z836xxzy).

**About Inference Hyperparameters:**
- **`cfg_text_scale`:** Controls how strongly the model follows the text prompt. `1.0` disables text guidance. Typical range: `4.0–8.0`.
- **`cfg_image_scale`:** Controls how much the model preserves input image details. `1.0` disables image guidance. Typical range: `1.0–2.0`.
- **`cfg_interval`:** Fraction of denoising steps where CFG is applied. Later steps can skip CFG to reduce computation. Typical: `[0.4, 1.0]`.
- **`timestep_shift`:** Shifts the distribution of denoising steps. Higher values allocate more steps at the start (affects layout); lower values allocate more at the end (improves details).
- **`num_timesteps`:** Total denoising steps. Typical: `50`.
- **`cfg_renorm_min`:** Minimum value for CFG-Renorm. `1.0` disables renorm. Typical: `0`.
- **`cfg_renorm_type`:** CFG-Renorm method:  
  - `global`: Normalize over all tokens and channels (default for T2I).
  - `channel`: Normalize across channels for each token.
  - `text_channel`: Like `channel`, but only applies to text condition (good for editing, may cause blur).
- **If edited images appear blurry, try `global` CFG-Renorm, decrease `cfg_renorm_min` or decrease `cfg_scale`.**


## 🔥 Quick Start

1️⃣  Set up environment
```bash
git clone https://github.com/bytedance-seed/BAGEL.git
cd BAGEL
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
```

2️⃣  Download pretrained checkpoint
```python
from huggingface_hub import snapshot_download

save_dir = "/path/to/save/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

```

3️⃣  Go to [`inference.ipynb`](inference.ipynb) to start playing with BAGEL!

4️⃣ Use Gradio WebUI to start playing with BAGEL!
```bash
pip install gradio
python app.py
```

## 🔥 Train & Eval

### Train

```bash
bash scripts/train.sh
```

You can replace the variables in the script with your own before running. 
See [TRAIN](TRAIN.md) for more details.

### Eval
We provide the scripts for evaluating VLM, T2I and Editing benchmarks. 
Please See [EVAL](EVAL.md) for more details.


## 📊 Benchmarks

### 1. Visual Understanding

| Model | MME ↑ | MMBench ↑ |   MMMU ↑ | MM-Vet ↑ | MathVista ↑ |
| ------------------- | ----------: | ----------: | -------: | -------: | ----------: |
| Janus-Pro-7B        | -  |     79.2 |     41.0 |     50.0 |           – |
| Qwen2.5-VL-7B      | 2347    |   83.5 | **58.6** |     67.1 |           68.2 |
| **BAGEL**    | **2388**  |  **85.0** |     55.3 | **67.2** |    **73.1** |

### 2. Text-to-Image Generation

| Model        | GenEval ↑ | WISE  ↑|
| ------------ | --------- | --------- |
| Janus-Pro-7B | 0.80      | 0.35 | 
| SD3-Medium   | 0.74      | - |
| FLUX-1-dev   | 0.82      | 0.50 |
| **BAGEL**    | -  | **0.52** |
| **BAGEL + CoT**    | **0.88**  | **0.70** |

### 3. Image Editing

| Model         | GEdit-Bench-EN (SC) ↑ | GEdit-Bench-EN (PQ) ↑ | GEdit-Bench-EN (O) ↑ | IntelligentBench ↑ |
| ------------- | --------------------- | --------------------- | ------------------- | ------------------ |
| Step1X-Edit   | 7.09                  | 6.76                  | **6.70**            | 14.9               |
| Gemini-2-exp. | 6.73                  | 6.61                  | 6.32                | **57.6**           |
| **BAGEL**     | **7.36**              | **6.83**              | 6.52                | 44.0               |
| **BAGEL+CoT** | –                   | –                     | –                   | 55.3               |


## ✍️ Citation

```bibtex
@article{deng2025bagel,
  title   = {Emerging Properties in Unified Multimodal Pretraining},
  author  = {Deng, Chaorui and Zhu, Deyao and Li, Kunchang and Gou, Chenhui and Li, Feng and Wang, Zeyu and Zhong, Shu and Yu, Weihao and Nie, Xiaonan and Song, Ziang and Shi, Guang and Fan, Haoqi},
  journal = {arXiv preprint arXiv:2505.14683},
  year    = {2025}
}
```


## 📜 License
BAGEL is licensed under the Apache 2.0.
