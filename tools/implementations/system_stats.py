import logging
import psutil
import pynvml
from tools.base import tool

logger = logging.getLogger(__name__)


@tool
def get_system_stats() -> str:
    """
    Returns real-time PC performance metrics: CPU load, RAM usage, GPU temperature,
    GPU utilization and VRAM usage.
    Use this when the user asks about PC performance, hardware load, lag, temperature,
    or memory issues.
    Example triggers: 'как грузится комп', 'температура видеокарты', 'сколько памяти',
    'почему лагает', 'сколько свободно оперативки', 'нагрузка на проц'.
    """
    try:
        cpu_pct = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / 1024**3
        ram_total_gb = ram.total / 1024**3

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            gpu_temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_gb = mem_info.used / 1024**3
            vram_total_gb = mem_info.total / 1024**3
            pynvml.nvmlShutdown()
            gpu_str = (
                f"GPU {gpu_name}: load {gpu_util}%, "
                f"temp {gpu_temp}°C, "
                f"VRAM {vram_used_gb:.1f}/{vram_total_gb:.1f} GB, "
                f"Free VRAM = {vram_total_gb - vram_used_gb:.1f} GB"
            )
        except pynvml.NVMLError as e:
            gpu_str = f"GPU info unavailable ({e})"

        result = (
            f"CPU: {cpu_pct}% | "
            f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f} GB ({ram.percent}%) | "
            f"{gpu_str}"
        )
        logger.info(f"Инструмент выполнен: get_system_stats → {result}")
        return result
    except Exception as e:
        logger.error(f"Ошибка get_system_stats: {e}")
        return f"Failed to get system stats: {str(e)}"
