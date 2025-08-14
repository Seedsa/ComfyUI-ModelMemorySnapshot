import os
import json
import shutil
from pathlib import Path
import logging

comfy_dir = Path(__file__).parent.parent.parent / "comfy"

model_management_path = str(comfy_dir / "model_management.py")
original_model_management_path = str(comfy_dir / "model_management_original.py")
is_patched = os.path.exists(original_model_management_path)

# 初始化全局缓存变量
_preloaded_models_cache = []
_preloaded_text_encoders_cache = []
_preloaded_diffusion_models_cache = []
_preloaded_vaes_cache = []
_initialization_done = False

def _apply_cuda_safe_patch():
    """Apply a permanent patch that avoid torch cuda init during snapshots"""

    shutil.copy(model_management_path, original_model_management_path)
    print(
        "[memory_snapshot_helper] ==> Applying CUDA-safe patch for model_management.py"
    )

    with open(model_management_path, "r") as f:
        content = f.read()

    # Find the get_torch_device function and modify the CUDA device access
    # The original line uses: return torch.device(torch.cuda.current_device())
    # We'll replace it with a check if CUDA is available

    # Define the patched content as a constant
    CUDA_SAFE_PATCH = """import os
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        else:
            logging.info("[memory_snapshot_helper] CUDA is not available, defaulting to cpu")
            return torch.device('cpu')  # Safe fallback during snapshot"""

    if "return torch.device(torch.cuda.current_device())" in content:
        patched_content = content.replace(
            "return torch.device(torch.cuda.current_device())", CUDA_SAFE_PATCH
        )

        # Save the patched version
        with open(model_management_path, "w") as f:
            f.write(patched_content)

        print("[memory_snapshot_helper] ==> Successfully patched model_management.py")
    else:
        raise Exception(
            "[memory_snapshot_helper] ==> Failed to patch model_management.py"
        )

def load_config():
    """加载JSON配置文件"""
    json_config_file = Path(__file__).parent / "preload_config.json"

    if not json_config_file.exists():
        logging.info(f"[memory_snapshot_helper] ==> JSON配置文件不存在: {json_config_file}")
        logging.info("[memory_snapshot_helper] ==> 请创建 preload_config.json 文件来配置要预加载的模型")
        return None

    try:
        with open(json_config_file, encoding="utf-8") as f:
            config = json.load(f)

        # 验证配置文件格式
        if "models" not in config:
            logging.info("[memory_snapshot_helper] ==> 无效的JSON配置: 缺少 'models' 部分")
            return None

        return config

    except json.JSONDecodeError as e:
        logging.info(f"[memory_snapshot_helper] ==> JSON配置文件解析失败: {e}")
        return None
    except Exception as e:
        logging.info(f"[memory_snapshot_helper] ==> 加载JSON配置文件出错: {e}")
        return None

def get_models_to_preload():
    """读取要预加载的检查点模型列表"""
    config = load_config()

    if not config:
        return []

    # 从JSON配置中读取
    checkpoints = config.get("models", {}).get("checkpoints", [])
    models = []
    for checkpoint in checkpoints:
        if isinstance(checkpoint, dict) and checkpoint.get("enabled", True):
            models.append(checkpoint["name"])
        elif isinstance(checkpoint, str):
            # 兼容简单字符串格式
            models.append(checkpoint)

    if models:
        logging.info(f"[memory_snapshot_helper] ==> 找到 {len(models)} 个检查点模型配置")

    return models

def get_text_encoders_to_preload():
    """读取要预加载的文本编码器列表"""
    config = load_config()

    if not config:
        return []

    # 从JSON配置中读取
    text_encoders_config = config.get("models", {}).get("text_encoders", [])
    text_encoders = []

    for encoder_config in text_encoders_config:
        if not isinstance(encoder_config, dict) or not encoder_config.get("enabled", True):
            continue

        clip_names = encoder_config.get("clip_names", [])
        if isinstance(clip_names, list):
            clip_names_str = ",".join(clip_names)
        else:
            clip_names_str = str(clip_names)

        encoder_type = encoder_config.get("type", "stable_diffusion")
        device = encoder_config.get("device", "default")

        # 构建配置字符串
        config_str = f"{clip_names_str}|{encoder_type}|{device}"
        text_encoders.append(config_str)

    if text_encoders:
        logging.info(f"[memory_snapshot_helper] ==> 找到 {len(text_encoders)} 个文本编码器配置")

    return text_encoders

def get_diffusion_models_to_preload():
    """读取要预加载的扩散模型列表"""
    config = load_config()

    if not config:
        return []

    # 从JSON配置中读取
    diffusion_models_config = config.get("models", {}).get("diffusion_models", [])
    diffusion_models = []

    for model_config in diffusion_models_config:
        if not isinstance(model_config, dict) or not model_config.get("enabled", True):
            continue

        model_name = model_config.get("model_name", "")
        weight_dtype = model_config.get("weight_dtype", "default")

        if model_name:
            # 构建配置字符串
            config_str = f"{model_name}|{weight_dtype}"
            diffusion_models.append(config_str)

    if diffusion_models:
        logging.info(f"[memory_snapshot_helper] ==> 找到 {len(diffusion_models)} 个扩散模型配置")

    return diffusion_models

def get_vaes_to_preload():
    """读取要预加载的VAE列表"""
    config = load_config()

    if not config:
        return []

    # 从JSON配置中读取
    vaes_config = config.get("models", {}).get("vaes", [])
    vaes = []

    for vae_config in vaes_config:
        if not isinstance(vae_config, dict) or not vae_config.get("enabled", True):
            continue

        vae_name = vae_config.get("vae_name", "")

        if vae_name:
            # 构建配置字符串
            vaes.append(vae_name)

    if vaes:
        logging.info(f"[memory_snapshot_helper] ==> 找到 {len(vaes)} 个VAE配置")

    return vaes


def preload_text_encoders():
    """预加载文本编码器"""
    text_encoders_to_preload = get_text_encoders_to_preload()
    if not text_encoders_to_preload:
        logging.info("[memory_snapshot_helper] ==> 没有配置要预加载的文本编码器")
        return

    try:
        import nodes
    except ImportError as e:
        logging.info(f"[memory_snapshot_helper] ==> Cannot import ComfyUI modules for text encoders: {e}")
        return

    # 使用全局存储预加载的文本编码器引用
    global _preloaded_text_encoders_cache

    for encoder_config in text_encoders_to_preload:
        try:
            # 解析配置格式
            parts = encoder_config.split("|")
            if len(parts) < 2:
                logging.info(f"[memory_snapshot_helper] ==> Invalid text encoder config format: {encoder_config}")
                continue

            clip_names = parts[0].strip()
            clip_type = parts[1].strip()
            device = parts[2].strip() if len(parts) > 2 else "default"

            # 分离多个编码器名称
            clip_name_list = [name.strip() for name in clip_names.split(",")]

            logging.info(f"[memory_snapshot_helper] ==> Preloading text encoder: {clip_names} (type: {clip_type})")

            # 根据编码器数量选择加载方式
            if len(clip_name_list) == 1:
                # 单个文本编码器
                clip_name = clip_name_list[0]
                clip = nodes.CLIPLoader().load_clip(clip_name, type=clip_type, device=device)[0]
                cache_key = f"{clip_name}_{clip_type}_{device}"
            elif len(clip_name_list) == 2:
                # 双文本编码器
                clip_name1, clip_name2 = clip_name_list
                clip = nodes.DualCLIPLoader().load_clip(clip_name1, clip_name2, type=clip_type, device=device)[0]
                cache_key = f"{clip_name1}_{clip_name2}_{clip_type}_{device}"
            elif len(clip_name_list) == 3:
                # 三重文本编码器
                clip_name1, clip_name2, clip_name3 = clip_name_list
                if "TripleCLIPLoader" in nodes.NODE_CLASS_MAPPINGS:
                    clip = nodes.NODE_CLASS_MAPPINGS["TripleCLIPLoader"]().load_clip(clip_name1, clip_name2, clip_name3)[0]
                    cache_key = f"{clip_name1}_{clip_name2}_{clip_name3}_{clip_type}_{device}"
                else:
                    logging.info(f"[memory_snapshot_helper] ==> TripleCLIPLoader not available, skipping: {encoder_config}")
                    continue
            else:
                logging.info(f"[memory_snapshot_helper] ==> Unsupported number of text encoders: {len(clip_name_list)}")
                continue

            # 存储到缓存
            _preloaded_text_encoders_cache.append({
                'key': cache_key,
                'clip_names': clip_names,
                'clip_type': clip_type,
                'device': device,
                'clip': clip,
                'config': encoder_config
            })

            logging.info(f"[memory_snapshot_helper] ==> Text encoder {clip_names} successfully preloaded and cached")

        except Exception as e:
            logging.info(f"[memory_snapshot_helper] ==> Failed to load text encoder {encoder_config}: {e}")
            import traceback
            traceback.print_exc()

    logging.info(f"[memory_snapshot_helper] ==> Text encoder preloading completed. {len(_preloaded_text_encoders_cache)} encoders cached.")

def preload_diffusion_models():
    """预加载扩散模型"""
    diffusion_models_to_preload = get_diffusion_models_to_preload()
    if not diffusion_models_to_preload:
        logging.info("[memory_snapshot_helper] ==> 没有配置要预加载的扩散模型")
        return

    try:
        import nodes
    except ImportError as e:
        logging.info(f"[memory_snapshot_helper] ==> Cannot import ComfyUI modules for diffusion models: {e}")
        return

    # 使用全局存储预加载的扩散模型引用
    global _preloaded_diffusion_models_cache

    for model_config in diffusion_models_to_preload:
        try:
            # 解析配置格式
            parts = model_config.split("|")
            model_name = parts[0].strip()
            if not model_name:
                logging.info(f"[memory_snapshot_helper] ==> Invalid diffusion model config format: {model_config}")
                continue

            weight_dtype = parts[1].strip() if len(parts) > 1 else "default"

            logging.info(f"[memory_snapshot_helper] ==> Preloading diffusion model: {model_name} (dtype: {weight_dtype})")

            # 加载扩散模型
            model = nodes.UNETLoader().load_unet(model_name, weight_dtype)[0]
            cache_key = f"{model_name}_{weight_dtype}"

            # 存储到缓存
            _preloaded_diffusion_models_cache.append({
                'key': cache_key,
                'model_name': model_name,
                'weight_dtype': weight_dtype,
                'model': model,
                'config': model_config
            })

            logging.info(f"[memory_snapshot_helper] ==> Diffusion model {model_name} successfully preloaded and cached")

        except Exception as e:
            logging.info(f"[memory_snapshot_helper] ==> Failed to load diffusion model {model_config}: {e}")
            import traceback
            traceback.print_exc()

    logging.info(f"[memory_snapshot_helper] ==> Diffusion model preloading completed. {len(_preloaded_diffusion_models_cache)} models cached.")

def preload_vaes():
    """预加载VAE模型"""
    vaes_to_preload = get_vaes_to_preload()
    if not vaes_to_preload:
        logging.info("[memory_snapshot_helper] ==> 没有配置要预加载的VAE模型")
        return

    try:
        import nodes
    except ImportError as e:
        logging.info(f"[memory_snapshot_helper] ==> Cannot import ComfyUI modules for VAEs: {e}")
        return

    # 使用全局存储预加载的VAE引用
    global _preloaded_vaes_cache

    for vae_name in vaes_to_preload:
        try:
            logging.info(f"[memory_snapshot_helper] ==> Preloading VAE: {vae_name}")

            # 加载VAE模型
            vae = nodes.VAELoader().load_vae(vae_name)[0]

            # 存储到缓存
            _preloaded_vaes_cache.append({
                'name': vae_name,
                'vae': vae
            })

            logging.info(f"[memory_snapshot_helper] ==> VAE {vae_name} successfully preloaded and cached")

        except Exception as e:
            logging.info(f"[memory_snapshot_helper] ==> Failed to load VAE {vae_name}: {e}")
            import traceback
            traceback.print_exc()

    logging.info(f"[memory_snapshot_helper] ==> VAE preloading completed. {len(_preloaded_vaes_cache)} VAEs cached.")


def preload_models():
    """按列表预加载模型并将其注册到ComfyUI的模型管理系统中"""
    global _initialization_done

    # 防止重复执行
    if _initialization_done:
        logging.info("[memory_snapshot_helper] ==> Already initialized, skipping preload.")
        return

    _initialization_done = True

    # 检查配置设置
    config = load_config()
    settings = config.get("settings", {}) if config else {}
    auto_gpu_load = settings.get("auto_gpu_load", True)

    models_to_preload = get_models_to_preload()
    if not models_to_preload:
        logging.info("[memory_snapshot_helper] ==> 没有配置要预加载的检查点模型")
    else:
        try:
            from comfy import model_management
            from comfy import sd
            import folder_paths
        except ImportError as e:
            logging.info(f"[memory_snapshot_helper] ==> Cannot import ComfyUI modules: {e}")
            return

        # 使用全局存储预加载的模型引用，防止被垃圾回收
        global _preloaded_models_cache

        for model_name in models_to_preload:
            try:
                logging.info(f"[memory_snapshot_helper] ==> Preloading checkpoint model: {model_name}")
                ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", model_name)

                # 加载模型
                model_patcher, clip, vae, clipvision = sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings")
                )

                if model_patcher is not None:
                    # 根据配置决定是否加载到GPU
                    if auto_gpu_load:
                        load_device = model_management.get_torch_device()
                        if not model_management.is_device_cpu(load_device):
                            logging.info(f"[memory_snapshot_helper] ==> Loading {model_name} to GPU for caching...")
                            # 使用ComfyUI的模型管理系统加载模型
                            model_management.load_model_gpu(model_patcher)
                            logging.info(f"[memory_snapshot_helper] ==> Model {model_name} cached in GPU memory")
                        else:
                            logging.info(f"[memory_snapshot_helper] ==> Model {model_name} loaded to CPU")
                    else:
                        logging.info(f"[memory_snapshot_helper] ==> Model {model_name} loaded (auto_gpu_load disabled)")

                    # 将模型引用存储在全局缓存中，防止被垃圾回收
                    _preloaded_models_cache.append({
                        'name': model_name,
                        'model_patcher': model_patcher,
                        'clip': clip,
                        'vae': vae,
                        'clipvision': clipvision,
                        'path': ckpt_path
                    })

                    logging.info(f"[memory_snapshot_helper] ==> Checkpoint model {model_name} successfully preloaded and cached")
                else:
                    logging.info(f"[memory_snapshot_helper] ==> Failed to create model patcher for {model_name}")

            except Exception as e:
                logging.info(f"[memory_snapshot_helper] ==> Failed to load checkpoint model {model_name}: {e}")
                import traceback
                traceback.print_exc()

        logging.info(f"[memory_snapshot_helper] ==> Checkpoint model preloading completed. {len(_preloaded_models_cache)} models cached.")

    # 预加载文本编码器
    preload_text_encoders()

    # 预加载扩散模型
    preload_diffusion_models()

    # 预加载VAE模型
    preload_vaes()



def get_preloaded_model(model_name):
    """从预加载缓存中获取模型"""
    global _preloaded_models_cache
    try:
        for cached_model in _preloaded_models_cache:
            if cached_model['name'] == model_name:
                return cached_model
    except (NameError, TypeError):
        pass
    return None

def is_model_preloaded(model_name):
    """检查模型是否已经预加载"""
    return get_preloaded_model(model_name) is not None

def get_preloaded_model_patcher(model_name):
    """获取预加载模型的ModelPatcher对象"""
    cached_model = get_preloaded_model(model_name)
    if cached_model:
        return cached_model.get('model_patcher')
    return None

def get_all_preloaded_models():
    """获取所有预加载的模型列表"""
    global _preloaded_models_cache
    try:
        return _preloaded_models_cache.copy()
    except (NameError, TypeError):
        return []

# === 文本编码器缓存访问函数 ===
def get_preloaded_text_encoder(cache_key):
    """从预加载缓存中获取文本编码器"""
    global _preloaded_text_encoders_cache
    try:
        for cached_encoder in _preloaded_text_encoders_cache:
            if cached_encoder['key'] == cache_key:
                return cached_encoder
    except (NameError, TypeError):
        pass
    return None

def is_text_encoder_preloaded(cache_key):
    """检查文本编码器是否已经预加载"""
    return get_preloaded_text_encoder(cache_key) is not None

def get_all_preloaded_text_encoders():
    """获取所有预加载的文本编码器列表"""
    global _preloaded_text_encoders_cache
    try:
        return _preloaded_text_encoders_cache.copy()
    except (NameError, TypeError):
        return []

# === 扩散模型缓存访问函数 ===
def get_preloaded_diffusion_model(cache_key):
    """从预加载缓存中获取扩散模型"""
    global _preloaded_diffusion_models_cache
    try:
        for cached_model in _preloaded_diffusion_models_cache:
            if cached_model['key'] == cache_key:
                return cached_model
    except (NameError, TypeError):
        pass
    return None

def is_diffusion_model_preloaded(cache_key):
    """检查扩散模型是否已经预加载"""
    return get_preloaded_diffusion_model(cache_key) is not None

def get_all_preloaded_diffusion_models():
    """获取所有预加载的扩散模型列表"""
    global _preloaded_diffusion_models_cache
    try:
        return _preloaded_diffusion_models_cache.copy()
    except (NameError, TypeError):
        return []

# === VAE缓存访问函数 ===
def get_preloaded_vae(vae_name):
    """从预加载缓存中获取VAE"""
    global _preloaded_vaes_cache
    try:
        for cached_vae in _preloaded_vaes_cache:
            if cached_vae['name'] == vae_name:
                return cached_vae
    except (NameError, TypeError):
        pass
    return None

def is_vae_preloaded(vae_name):
    """检查VAE是否已经预加载"""
    return get_preloaded_vae(vae_name) is not None

def get_all_preloaded_vaes():
    """获取所有预加载的VAE列表"""
    global _preloaded_vaes_cache
    try:
        return _preloaded_vaes_cache.copy()
    except (NameError, TypeError):
        return []


if not is_patched:
    _apply_cuda_safe_patch()

# 导出函数供其他模块使用
__all__ = [
    # 检查点模型缓存
    'get_preloaded_model', 'is_model_preloaded', 'get_preloaded_model_patcher', 'get_all_preloaded_models',
    # 文本编码器缓存
    'get_preloaded_text_encoder', 'is_text_encoder_preloaded', 'get_all_preloaded_text_encoders',
    # 扩散模型缓存
    'get_preloaded_diffusion_model', 'is_diffusion_model_preloaded', 'get_all_preloaded_diffusion_models',
    # VAE缓存
    'get_preloaded_vae', 'is_vae_preloaded', 'get_all_preloaded_vaes',
    "preload_models"
]
