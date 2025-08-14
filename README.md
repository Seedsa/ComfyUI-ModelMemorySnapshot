# ComfyUI Model Memory Snapshot - 模型预加载缓存系统

## 功能概述

ComfyUI Model Memory Snapshot 用于在 ComfyUI 启动时预加载和缓存模型，在 Serverless 场景下特别有用，配合类似 Modal 内存快照[Memory Snapshot](https://modal.com/docs/guide/memory-snapshot#memory-snapshot)等技术，大大提高工作流程的执行速度。支持以下模型类型：

- **检查点模型 (Checkpoints)** - 完整的 Stable Diffusion 模型
- **文本编码器 (Text Encoders)** - CLIP 模型，支持单个、双重、三重编码器
- **扩散模型 (Diffusion Models)** - UNet 模型
- **VAE 模型** - 变分自编码器

## 配置文件

系统使用 `preload_config.json` 作为唯一配置文件。配置示例：

```json
{
  "version": "1.0",
  "description": "ComfyUI 模型预加载配置文件",
  "models": {
    "checkpoints": [
      {
        "name": "dreamshaper_8.safetensors",
        "cache_key": "",
        "enabled": true,
        "description": "主要使用的检查点模型"
      }
    ],
    "text_encoders": [
      {
        "clip_names": ["clip_l.safetensors", "clip_g.safetensors"],
        "type": "sdxl",
        "device": "default",
        "cache_key": "",
        "enabled": true,
        "description": "SDXL双文本编码器"
      }
    ],
    "diffusion_models": [
      {
        "model_name": "flux1-dev.safetensors",
        "weight_dtype": "fp8_e4m3fn",
        "cache_key": "",
        "enabled": true,
        "description": "Flux扩散模型"
      }
    ],
    "vaes": [
      {
        "vae_name": "sdxl_vae.safetensors",
        "cache_key": "",
        "enabled": true,
        "description": "SDXL专用VAE"
      }
    ]
  },
  "settings": {
    "auto_gpu_load": true,
    "verbose": true
  }
}
```

## 可用节点

### 检查点模型

- **🚀 缓存检查点加载器** - 从缓存或直接加载检查点模型
- **🎯 预加载模型选择器** - 从预加载模型中选择

### 文本编码器

- **📝 缓存文本编码器加载器** - 支持单个/双重/三重文本编码器
- **📝 预加载文本编码器选择器** - 从预加载编码器中选择

### 扩散模型

- **🎨 缓存扩散模型加载器** - 加载 UNet 扩散模型
- **🎨 预加载扩散模型选择器** - 从预加载扩散模型中选择

### VAE 模型

- **🖼️ 缓存 VAE 加载器** - 从缓存或直接加载 VAE
- **🖼️ 预加载 VAE 选择器** - 从预加载 VAE 中选择

### 信息和状态

- **📋 缓存模型信息** - 显示所有预加载模型的详细信息
- **🔍 检查缓存状态** - 检查特定模型的缓存状态

## 使用流程

1. **配置模型**: 编辑 `preload_config.json`，启用需要预加载的模型
2. **启动 ComfyUI**: 模型会在启动时自动预加载到内存
3. **使用缓存节点**: 在工作流程中使用相应的缓存节点替代标准节点
4. **享受加速**: 预加载的模型会快速加载，显著提升工作效率

## 配置参数说明

### 通用参数

- `enabled`: 是否启用该配置项
- `cache_key`: 自定义缓存键（可选，留空自动生成）
- `description`: 配置描述（可选）

### 检查点模型

- `name`: 模型文件名（必需）

### 文本编码器

- `clip_names`: 编码器文件名列表（支持 1-3 个）
- `type`: 编码器类型（如 stable_diffusion, sdxl, sd3, flux 等）
- `device`: 设备类型（default 或 cpu）

### 扩散模型

- `model_name`: 模型文件名（必需）
- `weight_dtype`: 权重类型（default, fp8_e4m3fn, fp8_e4m3fn_fast, fp8_e5m2）

### VAE 模型

- `vae_name`: VAE 文件名（必需）

### 系统设置

- `auto_gpu_load`: 是否自动将模型加载到 GPU 内存
- `verbose`: 是否显示详细日志

## 性能优势

使用模型预加载缓存系统可以获得以下性能提升：

1. **启动时预加载**: 模型在 ComfyUI 启动时预加载，避免工作流程中的加载延迟
2. **内存缓存**: 预加载的模型保存在内存中，访问速度极快
3. **GPU 预热**: 模型可预先加载到 GPU 内存，避免运行时的 GPU 内存分配
4. **智能回退**: 如果模型未预加载，系统会自动回退到标准加载方式

## 注意事项

1. **内存占用**: 预加载会增加内存使用量，请确保有足够的系统内存
2. **启动时间**: 预加载过程会延长 ComfyUI 的启动时间
3. **模型路径**: 确保配置中的模型文件名与实际文件匹配
