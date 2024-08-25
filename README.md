# MyLLaVA
build LLaVA from scratch myself \n
TODO: multimodal group by length \n
llama3, mousi

Additional fix warning when new install environment:
# in deepspeed/runtime/zero/linear.py 36
import functools
# Fix `torch.[device].amp.custom_fwd/bwd` FutureWarning in torch 2.4
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'custom_fwd') and hasattr(torch.amp, 'custom_bwd'):
        autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=get_accelerator().device_name())
        autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=get_accelerator().device_name())
    else:
        # original implementation
        autocast_custom_fwd = get_accelerator().amp().custom_fwd
        autocast_custom_bwd = get_accelerator().amp().custom_bwd

# torch/utils/checkpoint.py 1399
    with device_autocast_ctx, torch.amp.autocast('cpu', **cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
        fn(*args, **kwargs)

# [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
# [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
https://blog.csdn.net/qq_51750957/article/details/129852539
conda install -c conda-forge gcc_linux-64
conda install -c conda-forge gxx_linux-64
Not helpful though

FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.

deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py 28
def load(self, path: str, map_location=None):
        logger.info(f"[Torch] Loading checkpoint from {path}...")
        partition = torch.load(path, weights_only=False, map_location=map_location)
        logger.info(f"[Torch] Loaded checkpoint from {path}.")
        return partition

transformers/trainer.py 2833
checkpoint_rng_state = torch.load(rng_file, weights_only=False)

# accelerate/big_modeling.py 493
if device != "disk":
    if device == "meta":
        model.to_empty(device)
    else:
        model.to(device)