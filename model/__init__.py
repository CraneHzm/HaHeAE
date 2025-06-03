from typing import Union
from .unet import BeatGANsUNetModel, BeatGANsUNetConfig, GCNUNetModel, GCNUNetConfig
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel, GCNAutoencConfig, GCNAutoencModel

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel, GCNUNetModel, GCNAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig, GCNUNetConfig,GCNAutoencConfig]