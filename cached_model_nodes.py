"""
缓存模型加载节点 - 用于加载预启动阶段缓存的模型
"""
import folder_paths
import logging
from .prestartup_script import (
    get_preloaded_model,
    is_model_preloaded,
    get_all_preloaded_models,
    # 文本编码器缓存
    get_preloaded_text_encoder,
    is_text_encoder_preloaded,
    get_all_preloaded_text_encoders,
    # 扩散模型缓存
    get_preloaded_diffusion_model,
    is_diffusion_model_preloaded,
    get_all_preloaded_diffusion_models,
    # VAE缓存
    get_preloaded_vae,
    is_vae_preloaded,
    get_all_preloaded_vaes,
    preload_models
)

preload_models()

class CachedCheckpointLoader:
    """从预加载缓存中加载检查点模型"""

    @classmethod
    def INPUT_TYPES(s):
        # 获取所有可用的检查点名称
        available_checkpoints = folder_paths.get_filename_list("checkpoints")

        return {
            "required": {
                "ckpt_name": (available_checkpoints, {"tooltip": "要加载的检查点模型名称"}),
                "use_cache": ("BOOLEAN", {"default": True, "label_off": "直接加载", "label_on": "使用缓存"}),
            },
            "optional": {
                "cache_key": ("STRING", {"multiline": False, "placeholder": "可选：指定缓存键名（留空则使用ckpt_name）"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("model", "clip", "vae", "cache_info", "from_cache")
    OUTPUT_TOOLTIPS = (
        "用于去噪潜在空间的扩散模型",
        "用于编码文本提示的CLIP模型",
        "用于编码和解码图像的VAE模型",
        "缓存状态信息",
        "是否从缓存加载"
    )

    FUNCTION = "load_checkpoint"
    CATEGORY = "MemorySnapshotHelper/loaders"
    DESCRIPTION = "从预加载缓存或直接加载检查点模型。如果模型已预加载到缓存中，将直接从缓存返回，大大提高加载速度。"

    def load_checkpoint(self, ckpt_name, use_cache=True, cache_key=""):
        # 确定要使用的缓存键
        key = cache_key.strip() if cache_key.strip() else ckpt_name

        # 检查是否使用缓存且模型已预加载
        if use_cache and is_model_preloaded(key):
            cached_model = get_preloaded_model(key)

            model_patcher = cached_model['model_patcher']
            clip = cached_model['clip']
            vae = cached_model['vae']

            cache_info = f"✅ 从缓存加载: {key}"
            from_cache = True

            logging.info(f"[CachedCheckpointLoader] ==> 从缓存加载模型: {key}")

        else:
            # 直接加载模型（原始方式）
            from comfy import sd

            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            out = sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )

            model_patcher, clip, vae = out[:3]

            if use_cache:
                cache_info = f"⚠️ 缓存未命中，直接加载: {ckpt_name} (未预加载: {key})"
                logging.info(f"[CachedCheckpointLoader] ==> 缓存未命中，直接加载: {ckpt_name}")
            else:
                cache_info = f"🔄 直接加载模式: {ckpt_name}"
                logging.info(f"[CachedCheckpointLoader] ==> 直接加载模式: {ckpt_name}")

            from_cache = False

        return model_patcher, clip, vae, cache_info, from_cache

    @staticmethod
    def IS_CHANGED(ckpt_name, use_cache=True, cache_key=""):
        # 如果使用缓存且模型已预加载，返回固定值避免重复计算
        key = cache_key.strip() if cache_key.strip() else ckpt_name
        if use_cache and is_model_preloaded(key):
            return f"cached_{key}"
        return ckpt_name


class CachedModelInfo:
    """显示缓存模型信息"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_info",)

    FUNCTION = "get_cache_info"
    CATEGORY = "MemorySnapshotHelper/info"
    OUTPUT_NODE = True
    DESCRIPTION = "显示当前预加载的模型缓存信息"

    def get_cache_info(self, unique_id):
        try:
            cached_models = get_all_preloaded_models()

            if not cached_models:
                info = "⚠️ 没有预加载的模型缓存"
            else:
                info_lines = ["📋 预加载模型缓存信息:", ""]

                for i, model in enumerate(cached_models, 1):
                    info_lines.append(f"{i}. 模型名称: {model['name']}")
                    info_lines.append(f"   路径: {model['path']}")
                    info_lines.append(f"   模型: {'✅' if model['model_patcher'] else '❌'}")
                    info_lines.append(f"   CLIP: {'✅' if model['clip'] else '❌'}")
                    info_lines.append(f"   VAE: {'✅' if model['vae'] else '❌'}")
                    info_lines.append("")

                info_lines.append(f"总计: {len(cached_models)} 个预加载模型")
                info = "\n".join(info_lines)

        except Exception as e:
            info = f"❌ 获取缓存信息失败: {str(e)}"

        logging.info(f"[CachedModelInfo] ==> 缓存信息: {len(get_all_preloaded_models())} 个模型")
        return (info,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 总是更新以显示最新信息
        return float("NaN")


class PreloadedModelSelector:
    """预加载模型选择器 - 只显示已预加载的模型"""

    @classmethod
    def INPUT_TYPES(s):
        # 获取预加载模型列表
        preloaded_models = [model['name'] for model in get_all_preloaded_models()]

        if not preloaded_models:
            preloaded_models = ["无预加载模型"]

        return {
            "required": {
                "model_name": (preloaded_models, {"tooltip": "选择一个预加载的模型"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name")
    OUTPUT_TOOLTIPS = (
        "预加载的扩散模型",
        "预加载的CLIP模型",
        "预加载的VAE模型",
        "模型名称"
    )

    FUNCTION = "load_preloaded_model"
    CATEGORY = "MemorySnapshotHelper/loaders"
    DESCRIPTION = "从预加载的模型缓存中选择并加载模型，确保快速访问。"

    def load_preloaded_model(self, model_name):
        if model_name == "无预加载模型":
            raise Exception("没有可用的预加载模型。请确保在预启动脚本中配置了要预加载的模型。")

        cached_model = get_preloaded_model(model_name)

        if cached_model is None:
            raise Exception(f"模型 '{model_name}' 未在缓存中找到")

        model_patcher = cached_model['model_patcher']
        clip = cached_model['clip']
        vae = cached_model['vae']

        logging.info(f"[PreloadedModelSelector] ==> 加载预缓存模型: {model_name}")

        return model_patcher, clip, vae, model_name

    @staticmethod
    def IS_CHANGED(model_name):
        # 对于预加载模型，返回固定值避免重复计算
        if model_name != "无预加载模型":
            return f"preloaded_{model_name}"
        return model_name


class CheckCacheStatus:
    """检查特定模型的缓存状态"""

    @classmethod
    def INPUT_TYPES(s):
        available_checkpoints = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "ckpt_name": (available_checkpoints, {"tooltip": "要检查的检查点模型名称"}),
            },
            "optional": {
                "cache_key": ("STRING", {"multiline": False, "placeholder": "可选：指定缓存键名"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_cached", "status_info")
    OUTPUT_TOOLTIPS = (
        "模型是否已缓存",
        "缓存状态信息"
    )

    FUNCTION = "check_cache_status"
    CATEGORY = "MemorySnapshotHelper/info"
    DESCRIPTION = "检查指定模型是否已预加载到缓存中"

    def check_cache_status(self, ckpt_name, cache_key=""):
        key = cache_key.strip() if cache_key.strip() else ckpt_name

        is_cached = is_model_preloaded(key)

        if is_cached:
            cached_model = get_preloaded_model(key)
            status_info = f"✅ 模型已缓存: {key}\n路径: {cached_model['path']}"
        else:
            status_info = f"❌ 模型未缓存: {key}"

        return is_cached, status_info

    @staticmethod
    def IS_CHANGED(ckpt_name, cache_key=""):
        key = cache_key.strip() if cache_key.strip() else ckpt_name
        return f"check_{key}"


class CachedTextEncoderLoader:
    """从预加载缓存中加载文本编码器"""

    @classmethod
    def INPUT_TYPES(s):
        available_text_encoders = folder_paths.get_filename_list("text_encoders")

        return {
            "required": {
                "clip_name1": (available_text_encoders, {"tooltip": "第一个文本编码器名称"}),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos","lumina2","wan","hidream","chroma","ace","omnigen2","qwen_image"], ),
                "use_cache": ("BOOLEAN", {"default": True, "label_off": "直接加载", "label_on": "使用缓存"}),
            },
            "optional": {
                "clip_name2": (["None"] + available_text_encoders, ),
                "clip_name3": (["None"] + available_text_encoders, ),
                "device": (["default", "cpu"], {"advanced": True}),
                "cache_key": ("STRING", {"multiline": False, "placeholder": "可选：指定缓存键名"}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING", "BOOLEAN")
    RETURN_NAMES = ("clip", "cache_info", "from_cache")
    OUTPUT_TOOLTIPS = (
        "加载的CLIP模型",
        "缓存状态信息",
        "是否从缓存加载"
    )

    FUNCTION = "load_text_encoder"
    CATEGORY = "MemorySnapshotHelper/loaders"
    DESCRIPTION = "从预加载缓存或直接加载文本编码器。支持单个、双重或三重文本编码器。"

    def load_text_encoder(self, clip_name1, type, use_cache=True, clip_name2="None", clip_name3="None", device="default", cache_key=""):
        # 构建缓存键
        if cache_key.strip():
            key = cache_key.strip()
        else:
            key = clip_name1
            if clip_name2 != "None":
                key += f"_{clip_name2}"
            if clip_name3 != "None":
                key += f"_{clip_name3}"
            key += f"_{type}_{device}"

        # 检查是否使用缓存且编码器已预加载
        if use_cache and is_text_encoder_preloaded(key):
            cached_encoder = get_preloaded_text_encoder(key)
            clip = cached_encoder['clip']

            cache_info = f"✅ 从缓存加载文本编码器: {key}"
            from_cache = True

            logging.info(f"[CachedTextEncoderLoader] ==> 从缓存加载文本编码器: {key}")

        else:
            # 直接加载编码器
            import nodes

            # 根据编码器数量选择加载方式
            clip_name_list = [name for name in [clip_name1, clip_name2, clip_name3] if name != "None"]

            if len(clip_name_list) == 1:
                clip = nodes.CLIPLoader().load_clip(clip_name1, type=type, device=device)[0]
            elif len(clip_name_list) == 2:
                clip = nodes.DualCLIPLoader().load_clip(clip_name1, clip_name_list[1], type=type, device=device)[0]
            elif len(clip_name_list) == 3:
                if "TripleCLIPLoader" in nodes.NODE_CLASS_MAPPINGS:
                    clip = nodes.NODE_CLASS_MAPPINGS["TripleCLIPLoader"]().load_clip(clip_name1, clip_name_list[1], clip_name_list[2])[0]
                else:
                    raise Exception("TripleCLIPLoader 不可用")
            else:
                raise Exception(f"不支持的文本编码器数量: {len(clip_name_list)}")

            if use_cache:
                cache_info = f"⚠️ 缓存未命中，直接加载: {key}"
                logging.info(f"[CachedTextEncoderLoader] ==> 缓存未命中，直接加载: {key}")
            else:
                cache_info = f"🔄 直接加载模式: {key}"
                logging.info(f"[CachedTextEncoderLoader] ==> 直接加载模式: {key}")

            from_cache = False

        return clip, cache_info, from_cache

    @staticmethod
    def IS_CHANGED(clip_name1, type, use_cache=True, clip_name2="None", clip_name3="None", device="default", cache_key=""):
        # 构建缓存键
        if cache_key.strip():
            key = cache_key.strip()
        else:
            key = clip_name1
            if clip_name2 != "None":
                key += f"_{clip_name2}"
            if clip_name3 != "None":
                key += f"_{clip_name3}"
            key += f"_{type}_{device}"

        if use_cache and is_text_encoder_preloaded(key):
            return f"cached_te_{key}"
        return f"te_{clip_name1}_{clip_name2}_{clip_name3}_{type}_{device}"


class CachedDiffusionModelLoader:
    """从预加载缓存中加载扩散模型"""

    @classmethod
    def INPUT_TYPES(s):
        available_models = folder_paths.get_filename_list("diffusion_models")

        return {
            "required": {
                "model_name": (available_models, {"tooltip": "扩散模型名称"}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "use_cache": ("BOOLEAN", {"default": True, "label_off": "直接加载", "label_on": "使用缓存"}),
            },
            "optional": {
                "cache_key": ("STRING", {"multiline": False, "placeholder": "可选：指定缓存键名"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "BOOLEAN")
    RETURN_NAMES = ("model", "cache_info", "from_cache")
    OUTPUT_TOOLTIPS = (
        "加载的扩散模型",
        "缓存状态信息",
        "是否从缓存加载"
    )

    FUNCTION = "load_diffusion_model"
    CATEGORY = "MemorySnapshotHelper/loaders"
    DESCRIPTION = "从预加载缓存或直接加载扩散模型。"

    def load_diffusion_model(self, model_name, weight_dtype, use_cache=True, cache_key=""):
        # 构建缓存键
        key = cache_key.strip() if cache_key.strip() else f"{model_name}_{weight_dtype}"

        # 检查是否使用缓存且模型已预加载
        if use_cache and is_diffusion_model_preloaded(key):
            cached_model = get_preloaded_diffusion_model(key)
            model = cached_model['model']

            cache_info = f"✅ 从缓存加载扩散模型: {key}"
            from_cache = True

            logging.info(f"[CachedDiffusionModelLoader] ==> 从缓存加载扩散模型: {key}")

        else:
            # 直接加载模型
            import nodes

            model = nodes.UNETLoader().load_unet(model_name, weight_dtype)[0]

            if use_cache:
                cache_info = f"⚠️ 缓存未命中，直接加载: {model_name} (未预加载: {key})"
                logging.info(f"[CachedDiffusionModelLoader] ==> 缓存未命中，直接加载: {model_name}")
            else:
                cache_info = f"🔄 直接加载模式: {model_name}"
                logging.info(f"[CachedDiffusionModelLoader] ==> 直接加载模式: {model_name}")

            from_cache = False

        return model, cache_info, from_cache

    @staticmethod
    def IS_CHANGED(model_name, weight_dtype, use_cache=True, cache_key=""):
        key = cache_key.strip() if cache_key.strip() else f"{model_name}_{weight_dtype}"
        if use_cache and is_diffusion_model_preloaded(key):
            return f"cached_dm_{key}"
        return f"dm_{model_name}_{weight_dtype}"


class PreloadedTextEncoderSelector:
    """预加载文本编码器选择器"""

    @classmethod
    def INPUT_TYPES(s):
        preloaded_encoders = get_all_preloaded_text_encoders()
        encoder_keys = [encoder['key'] for encoder in preloaded_encoders]

        if not encoder_keys:
            encoder_keys = ["无预加载文本编码器"]

        return {
            "required": {
                "encoder_key": (encoder_keys, {"tooltip": "选择一个预加载的文本编码器"}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "encoder_info")
    OUTPUT_TOOLTIPS = (
        "预加载的CLIP模型",
        "编码器信息"
    )

    FUNCTION = "load_preloaded_text_encoder"
    CATEGORY = "MemorySnapshotHelper/loaders"
    DESCRIPTION = "从预加载的文本编码器缓存中选择并加载编码器。"

    def load_preloaded_text_encoder(self, encoder_key):
        if encoder_key == "无预加载文本编码器":
            raise Exception("没有可用的预加载文本编码器。请确保在预启动脚本中配置了要预加载的文本编码器。")

        cached_encoder = get_preloaded_text_encoder(encoder_key)

        if cached_encoder is None:
            raise Exception(f"文本编码器 '{encoder_key}' 未在缓存中找到")

        clip = cached_encoder['clip']
        encoder_info = f"类型: {cached_encoder['clip_type']}, 设备: {cached_encoder['device']}, 名称: {cached_encoder['clip_names']}"

        logging.info(f"[PreloadedTextEncoderSelector] ==> 加载预缓存文本编码器: {encoder_key}")

        return clip, encoder_info

    @staticmethod
    def IS_CHANGED(encoder_key):
        if encoder_key != "无预加载文本编码器":
            return f"preloaded_te_{encoder_key}"
        return encoder_key


class PreloadedDiffusionModelSelector:
    """预加载扩散模型选择器"""

    @classmethod
    def INPUT_TYPES(s):
        preloaded_models = get_all_preloaded_diffusion_models()
        model_keys = [model['key'] for model in preloaded_models]

        if not model_keys:
            model_keys = ["无预加载扩散模型"]

        return {
            "required": {
                "model_key": (model_keys, {"tooltip": "选择一个预加载的扩散模型"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    OUTPUT_TOOLTIPS = (
        "预加载的扩散模型",
        "模型信息"
    )

    FUNCTION = "load_preloaded_diffusion_model"
    CATEGORY = "MemorySnapshotHelper/loaders"
    DESCRIPTION = "从预加载的扩散模型缓存中选择并加载模型。"

    def load_preloaded_diffusion_model(self, model_key):
        if model_key == "无预加载扩散模型":
            raise Exception("没有可用的预加载扩散模型。请确保在预启动脚本中配置了要预加载的扩散模型。")

        cached_model = get_preloaded_diffusion_model(model_key)

        if cached_model is None:
            raise Exception(f"扩散模型 '{model_key}' 未在缓存中找到")

        model = cached_model['model']
        model_info = f"模型名称: {cached_model['model_name']}, 权重类型: {cached_model['weight_dtype']}"

        logging.info(f"[PreloadedDiffusionModelSelector] ==> 加载预缓存扩散模型: {model_key}")

        return model, model_info

    @staticmethod
    def IS_CHANGED(model_key):
        if model_key != "无预加载扩散模型":
            return f"preloaded_dm_{model_key}"
        return model_key


class CachedVAELoader:
    """从预加载缓存中加载VAE"""

    @classmethod
    def INPUT_TYPES(s):
        available_vaes = folder_paths.get_filename_list("vae")

        return {
            "required": {
                "vae_name": (available_vaes, {"tooltip": "VAE模型名称"}),
                "use_cache": ("BOOLEAN", {"default": True, "label_off": "直接加载", "label_on": "使用缓存"}),
            },
            "optional": {
                "cache_key": ("STRING", {"multiline": False, "placeholder": "可选：指定缓存键名"}),
            }
        }

    RETURN_TYPES = ("VAE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("vae", "cache_info", "from_cache")
    OUTPUT_TOOLTIPS = (
        "加载的VAE模型",
        "缓存状态信息",
        "是否从缓存加载"
    )

    FUNCTION = "load_vae"
    CATEGORY = "MemorySnapshotHelper/loaders"
    DESCRIPTION = "从预加载缓存或直接加载VAE模型。"

    def load_vae(self, vae_name, use_cache=True, cache_key=""):
        # 构建缓存键
        key = cache_key.strip() if cache_key.strip() else vae_name

        # 检查是否使用缓存且VAE已预加载
        if use_cache and is_vae_preloaded(key):
            cached_vae = get_preloaded_vae(key)
            vae = cached_vae['vae']

            cache_info = f"✅ 从缓存加载VAE: {key}"
            from_cache = True

            logging.info(f"[CachedVAELoader] ==> 从缓存加载VAE: {key}")

        else:
            # 直接加载VAE
            import nodes

            vae = nodes.VAELoader().load_vae(vae_name)[0]

            if use_cache:
                cache_info = f"⚠️ 缓存未命中，直接加载: {vae_name} (未预加载: {key})"
                logging.info(f"[CachedVAELoader] ==> 缓存未命中，直接加载: {vae_name}")
            else:
                cache_info = f"🔄 直接加载模式: {vae_name}"
                logging.info(f"[CachedVAELoader] ==> 直接加载模式: {vae_name}")

            from_cache = False

        return vae, cache_info, from_cache

    @staticmethod
    def IS_CHANGED(vae_name, use_cache=True, cache_key=""):
        key = cache_key.strip() if cache_key.strip() else vae_name
        if use_cache and is_vae_preloaded(key):
            return f"cached_vae_{key}"
        return f"vae_{vae_name}"


class PreloadedVAESelector:
    """预加载VAE选择器"""

    @classmethod
    def INPUT_TYPES(s):
        preloaded_vaes = get_all_preloaded_vaes()
        vae_names = [vae['name'] for vae in preloaded_vaes]

        if not vae_names:
            vae_names = ["无预加载VAE"]

        return {
            "required": {
                "vae_name": (vae_names, {"tooltip": "选择一个预加载的VAE"}),
            }
        }

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "vae_info")
    OUTPUT_TOOLTIPS = (
        "预加载的VAE模型",
        "VAE信息"
    )

    FUNCTION = "load_preloaded_vae"
    CATEGORY = "MemorySnapshotHelper/loaders"
    DESCRIPTION = "从预加载的VAE缓存中选择并加载VAE。"

    def load_preloaded_vae(self, vae_name):
        if vae_name == "无预加载VAE":
            raise Exception("没有可用的预加载VAE。请确保在预启动脚本中配置了要预加载的VAE。")

        cached_vae = get_preloaded_vae(vae_name)

        if cached_vae is None:
            raise Exception(f"VAE '{vae_name}' 未在缓存中找到")

        vae = cached_vae['vae']
        vae_info = f"VAE名称: {cached_vae['name']}"

        logging.info(f"[PreloadedVAESelector] ==> 加载预缓存VAE: {vae_name}")

        return vae, vae_info

    @staticmethod
    def IS_CHANGED(vae_name):
        if vae_name != "无预加载VAE":
            return f"preloaded_vae_{vae_name}"
        return vae_name


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "CachedCheckpointLoader": CachedCheckpointLoader,
    "CachedModelInfo": CachedModelInfo,
    "PreloadedModelSelector": PreloadedModelSelector,
    "CheckCacheStatus": CheckCacheStatus,
    "CachedTextEncoderLoader": CachedTextEncoderLoader,
    "CachedDiffusionModelLoader": CachedDiffusionModelLoader,
    "PreloadedTextEncoderSelector": PreloadedTextEncoderSelector,
    "PreloadedDiffusionModelSelector": PreloadedDiffusionModelSelector,
    "CachedVAELoader": CachedVAELoader,
    "PreloadedVAESelector": PreloadedVAESelector,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "CachedCheckpointLoader": "🚀 缓存检查点加载器",
    "CachedModelInfo": "📋 缓存模型信息",
    "PreloadedModelSelector": "🎯 预加载模型选择器",
    "CheckCacheStatus": "🔍 检查缓存状态",
    "CachedTextEncoderLoader": "📝 缓存文本编码器加载器",
    "CachedDiffusionModelLoader": "🎨 缓存扩散模型加载器",
    "PreloadedTextEncoderSelector": "📝 预加载文本编码器选择器",
    "PreloadedDiffusionModelSelector": "🎨 预加载扩散模型选择器",
    "CachedVAELoader": "🖼️ 缓存VAE加载器",
    "PreloadedVAESelector": "🖼️ 预加载VAE选择器",
}
