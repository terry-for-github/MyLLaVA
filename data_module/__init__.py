from .dataset import LazySingleImageAtFirstDialogDataset
from .image_loader import ImageLoader
from .data_collator import DataCollatorForSingleImageAtFirstDialog


def get_dataset_and_data_collator(tokenizer, data_args, vision_model_name,
                                  vision_token_num, version):
    image_loader = ImageLoader(
        image_folder=data_args.image_folder,
        vision_model_name=vision_model_name,
        image_mark=data_args.image_mark,
        image_process_mode=data_args.image_process_mode)

    return (
        LazySingleImageAtFirstDialogDataset(
            data_args=data_args,
            image_loader=image_loader,
            vision_token_num=vision_token_num,
            model_max_length=tokenizer.model_max_length,
        ),
        DataCollatorForSingleImageAtFirstDialog(
            tokenizer=tokenizer,
            version=version,
            image_mark=data_args.image_mark,
        )
    )


__all__ = ['get_dataset_and_data_collator']
