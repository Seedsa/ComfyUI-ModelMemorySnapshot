"""
Memory Snapshot Helper - ComfyUI 模型缓存系统
支持预启动阶段预加载模型，并在工作流中快速加载缓存的模型
"""

from .cached_model_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from server import PromptServer
import os
from aiohttp import web

# ------- API Endpoints -------


@PromptServer.instance.routes.post("/cuda/set_device")
async def set_current_device(request):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return web.json_response({"status": "success"})


# 导出节点映射
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("[memory_snapshot_helper] ==> 缓存模型加载节点已注册")
