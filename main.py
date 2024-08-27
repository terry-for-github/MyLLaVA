import builtins
import os

from llavax.train.train import build_and_train


def set_logger(is_main_process: bool):
    if not is_main_process:
        return
    import logging
    from datetime import datetime
    from transformers import logging as transformers_logging
    from deepspeed import logger as deepspeed_logger

    transformers_logging.set_verbosity_info()
    deepspeed_logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/log_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    transformers_logger = logging.getLogger('transformers')
    for handler in transformers_logger.handlers[:]:
        transformers_logger.removeHandler(handler)
    for handlder in deepspeed_logger.handlers[:]:
        deepspeed_logger.removeHandler(handlder)
    transformers_logger.propagate = False
    deepspeed_logger.propagate = False

    info_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
    info_handler.setLevel(logging.INFO)

    warning_handler = logging.FileHandler(os.path.join(log_dir, 'warn.log'))
    warning_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)

    transformers_logger.addHandler(info_handler)
    transformers_logger.addHandler(warning_handler)
    deepspeed_logger.addHandler(info_handler)
    deepspeed_logger.addHandler(warning_handler)


def set_builtin_print(is_local_main_process: bool):
    builtins_print = builtins.print

    if 'LLAVA_DEBUG' not in os.environ:
        debug_level = 0
    else:
        debug_level = int(os.environ['LLAVA_DEBUG'])
    assert debug_level >= 0

    def custom_print(*args, rank0_only=True, **kwargs):
        if len(args) == 0:
            builtins_print(**kwargs)
            return
        now_debug_level = 0
        if isinstance(args[0], str) and args[0].startswith('[DEBUG]'):
            assert isinstance(args[1], int) and 1 <= args[1] <= 9
            now_debug_level = args[1]
        if debug_level < now_debug_level:
            return
        if not is_local_main_process and rank0_only:
            return
        builtins_print(*args, **kwargs)

    builtins.print = custom_print


if __name__ == '__main__':
    from accelerate import PartialState
    is_main_process = PartialState().is_main_process
    is_local_main_process = PartialState().is_local_main_process

    set_builtin_print(is_local_main_process)

    set_logger(is_main_process)

    build_and_train()
