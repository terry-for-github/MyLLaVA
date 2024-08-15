from .dataset import LazyMMDialogDataset
from .data_collator import DataCollator
from .image_loader import ImageLoader


def get_dataset(data_args, image_processor):
    image_loader = ImageLoader(
        image_folder=data_args.image_folder,
        image_processor=image_processor,
        image_mark=data_args.image_mark,
        image_process_mode=data_args.image_process_mode)

    return LazyMMDialogDataset(
        json_path=data_args.json_path,
        image_loader=image_loader,
        dialog_key=data_args.dialog_key,
        image_key=data_args.image_key,
        role_key=data_args.role_key,
        content_key=data_args.content_key,
        human_key=data_args.human_key,
        gpt_key=data_args.gpt_key)


__all__ = ['get_dataset', 'DataCollator']
