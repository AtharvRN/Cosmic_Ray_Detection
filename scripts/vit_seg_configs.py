import ml_collections


def get_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 60
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 64
    config.transformer.num_heads = 6
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.patch_size = 16

    # config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 1
    config.activation = 'softmax'

    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (1,1,1)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    # config.decoder_channels = (128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    # config.skip_channels = [256, 64, 16]
    config.n_classes = 1
    config.n_skip = 3
    config.activation = 'softmax'

    return config
