import logging

logger = logging.getLogger(__name__)

def get_device() -> dict:
    """
    Detects available hardware accelerators safely and returns the best available device.
    Priority:
    1. NVIDIA CUDA
    2. Intel GPU / NPU via OpenVINO
    3. DirectML (AMD/Intel on Windows)
    4. CPU fallback
    
    Returns:
        dict: {"device": "...", "backend": "..."}
    """
    
    # 1. Check for CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Hardware Detected: NVIDIA CUDA")
            return {"device": "cuda", "backend": "cuda"}
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")

    # 2. Check for OpenVINO (Intel Integrated Graphics / NPU)
    try:
        from openvino.runtime import Core
        core = Core()
        available_devices = core.available_devices
        if 'GPU' in available_devices:
            logger.info("Hardware Detected: Intel GPU via OpenVINO")
            return {"device": "intel_gpu", "backend": "openvino"}
        elif 'NPU' in available_devices:
             logger.info("Hardware Detected: Intel NPU via OpenVINO")
             return {"device": "intel_npu", "backend": "openvino"}
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"OpenVINO detection failed: {e}")

    # 3. Check for DirectML (Windows Generic Hardware Acceleration)
    try:
        import torch_directml
        if torch_directml.is_available():
            logger.info("Hardware Detected: DirectML (Generic GPU)")
            return {"device": "directml", "backend": "directml"}
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"DirectML detection failed: {e}")

    # 4. CPU Fallback
    logger.info("Hardware Detected: CPU (No dedicated acceleration found)")
    return {"device": "cpu", "backend": "cpu"}
