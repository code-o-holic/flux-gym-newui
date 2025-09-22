import os
import time
from typing import Optional

import psutil

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

from torch.utils.tensorboard import SummaryWriter  # type: ignore


def init_nvml() -> None:
    if _HAS_NVML:
        try:
            pynvml.nvmlInit()
        except Exception:
            pass


def close_nvml() -> None:
    if _HAS_NVML:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def log_system_metrics(log_dir: str, interval_sec: int = 1, max_seconds: Optional[int] = None) -> None:
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    init_nvml()
    start = time.time()
    step = 0
    try:
        while True:
            now = time.time()
            if max_seconds is not None and now - start > max_seconds:
                break

            cpu_percent = psutil.cpu_percent(interval=None)
            virtual_mem = psutil.virtual_memory()
            writer.add_scalar('system/cpu_percent', cpu_percent, step)
            writer.add_scalar('system/ram_used_gb', virtual_mem.used / (1024 ** 3), step)
            writer.add_scalar('system/ram_total_gb', virtual_mem.total / (1024 ** 3), step)

            if _HAS_NVML:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        writer.add_scalar(f'gpu/{i}_util_percent', util.gpu, step)
                        writer.add_scalar(f'gpu/{i}_mem_used_gb', mem.used / (1024 ** 3), step)
                        writer.add_scalar(f'gpu/{i}_mem_total_gb', mem.total / (1024 ** 3), step)
                except Exception:
                    pass

            writer.flush()
            step += 1
            time.sleep(interval_sec)
    finally:
        close_nvml()
        writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--interval', type=int, default=1)
    args = parser.parse_args()
    log_system_metrics(args.log_dir, interval_sec=args.interval)


