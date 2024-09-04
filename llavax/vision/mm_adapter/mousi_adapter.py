import torch
import torch.nn as nn
import torch.nn.functional as F


class MousiAdapter(nn.Module):
    def __init__(
        self,
        mm_hidden_size_list: list[int],
        m_patch_one_token: list[int],
        llm_hidden_size: int
    ):
        super().__init__()

        self.mlp1_list = nn.ModuleList()
        self.m_list = m_patch_one_token
        for i, mm_hidden_size in enumerate(mm_hidden_size_list):
            self.mlp1_list.append(nn.Linear(mm_hidden_size * self.m_list[i], llm_hidden_size))
        self.mlp2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, image_features_list: list[torch.Tensor]):
        hidden_features_list = []
        # print('projector before:', list(map(lambda x: x.shape, image_features_list)))
        for i, image_features in enumerate(image_features_list):
            # m-patches-one-token
            # image_features: bs, num_patches, hidden_size
            # graph_encoder'm must be 1
            if isinstance(image_features, torch.Tensor) and self.m_list[i] > 1:
                bs, num_patches, _ = image_features.shape
                image_features = image_features.view(bs, num_patches // self.m_list[i], -1)
            hidden_features_list.append(self.mlp1_list[i](image_features))
        # print('projector after:', list(map(lambda x: x.shape, hidden_features_list)))
        hidden_feature = torch.cat(hidden_features_list, dim=1)
        # print('projector final:', hidden_feature.shape)
        return self.mlp2(F.gelu(hidden_feature))


# class MousiProjectorV2(nn.Module):
#     def __init__(self, image_hidden_size_list: List[int], m_token_one_patch: List[int], llm_hidden_size: int):
#         super().__init__()

#         self.mlp1_list = nn.ModuleList()
#         self.mlp2_list = nn.ModuleList()
#         self.m_list = m_token_one_patch
#         for i, image_hidden_size in enumerate(image_hidden_size_list):
#             # special judge for sg encoder
#             if image_hidden_size == 'sg_size':
#                 self.mlp1_list.append(SGAdapter())
#                 self.mlp2_list.append(nn.Sequential())
#             else:
#                 self.mlp1_list.append(nn.Linear(image_hidden_size * self.m_list[i], llm_hidden_size))
#                 self.mlp2_list.append(nn.Linear(llm_hidden_size, llm_hidden_size))

#     def forward(self, image_features_list: List[torch.Tensor]):
#         hidden_features_list = []
#         # print('projector before:', list(map(lambda x: x.shape, image_features_list)))
#         for i, image_features in enumerate(image_features_list):
#             # m-patches-one-token
#             # image_features: bs, num_patches, hidden_size
#             # graph_encoder'm must be 1
#             if isinstance(image_features, torch.Tensor) and self.m_list[i] > 1:
#                 bs, num_patches, _ = image_features.shape
#                 image_features = image_features.view(bs, num_patches // self.m_list[i], -1)
#             mlp1_features = self.mlp1_list[i](image_features)
#             gelu_features = F.gelu(mlp1_features)
#             mlp2_features = self.mlp2_list[i](gelu_features)
#             hidden_features_list.append(mlp2_features)
#         # print('projector after:', list(map(lambda x: x.shape, hidden_features_list)))
#         hidden_feature = torch.cat(hidden_features_list, dim=1)
#         # print('projector final:', hidden_feature.shape)
#         return hidden_feature
