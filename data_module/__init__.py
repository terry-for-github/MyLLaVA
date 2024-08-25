from .dataset import LazySingleImageAtFirstDialogDataset
from .image_loader import ImageLoader, MultiTowersImageLoader
from .data_collator import DataCollatorForSingleImageAtFirstDialog


def get_dataset_and_data_collator(tokenizer, data_args, vision_tower,
                                  num_vision_token, version):
    if len(vision_tower) > 1:
        image_loader = MultiTowersImageLoader(
            image_folder=data_args.image_folder,
            vision_model_list=vision_tower,
            image_process_mode=data_args.image_process_mode)
    else:
        image_loader = ImageLoader(
            image_folder=data_args.image_folder,
            vision_model_name=vision_tower[0],
            image_process_mode=data_args.image_process_mode)

    return (
        LazySingleImageAtFirstDialogDataset(
            data_args=data_args,
            image_loader=image_loader,
            vision_token_num=num_vision_token,
            model_max_length=tokenizer.model_max_length,
        ),
        DataCollatorForSingleImageAtFirstDialog(
            tokenizer=tokenizer,
            version=version,
        )
    )


__all__ = ['get_dataset_and_data_collator']
