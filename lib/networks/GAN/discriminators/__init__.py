
from lib.networks.GAN.discriminators import conv

discriminator_dict = {
    'dc': conv.DCDiscriminator,
    'resnet': conv.DiscriminatorResnet,
}