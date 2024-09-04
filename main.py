import builtins
import os

import transformers
from llavax.arguments import ModelArguments, DataArguments, TrainingArguments
from llavax.train import build_and_train


def config_logger():
    import logging
    import glob
    from datetime import datetime
    from transformers import logging as transformers_logging
    from deepspeed import logger as deepspeed_logger
    rank = os.environ.get('RANK', '-1')

    def _add_file_handler(logger: logging.Logger, log_dir: str, level=logging.INFO):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        level_name = logging.getLevelName(level)
        log_filename = os.path.join(log_dir, f'{logger.name}_{level_name}_{rank}.log')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    transformers_logging.set_verbosity_info()
    transformers_logger = logging.getLogger('transformers')
    deepspeed_logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = f"logs/log_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    for handler in transformers_logger.handlers:
        transformers_logger.removeHandler(handler)
    for handlder in deepspeed_logger.handlers:
        deepspeed_logger.removeHandler(handlder)
    transformers_logger.propagate = False
    deepspeed_logger.propagate = False

    _add_file_handler(transformers_logger, log_dir, level=logging.INFO)
    _add_file_handler(transformers_logger, log_dir, level=logging.WARNING)
    _add_file_handler(deepspeed_logger, log_dir, level=logging.INFO)
    _add_file_handler(deepspeed_logger, log_dir, level=logging.WARNING)

    if rank in ['0', '-1']:
        import shutil
        log_files = sorted(glob.glob(os.path.join('logs', "log_*")), key=os.path.getctime)

        while len(log_files) > 10:
            oldest_log = log_files.pop(0)
            shutil.rmtree(oldest_log)


def set_builtin_print(is_local_main_process: bool):
    builtins_print = builtins.print

    if 'LLAVA_DEBUG' not in os.environ:
        debug_level = 0
    else:
        debug_level = int(os.environ['LLAVA_DEBUG'])
    assert debug_level >= 0

    rank = os.environ.get('RANK', '-1')

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
        builtins_print(f'[RANK{rank}]', *args, **kwargs)

    builtins.print = custom_print


if __name__ == '__main__':
    from accelerate import PartialState
    is_main_process = PartialState().is_main_process
    is_local_main_process = PartialState().is_local_main_process

    set_builtin_print(is_local_main_process)

    config_logger()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)  # type: ignore
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print('[DEBUG]', 1, '===================================================================')
    print('[DEBUG]', 1, model_args)
    print('[DEBUG]', 1, data_args)
    print('[DEBUG]', 1, training_args)
    print('[DEBUG]', 1, '===================================================================')
    build_and_train(model_args, data_args, training_args)
