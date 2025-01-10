from torch.nn import ConvTranspose3d, Conv3d, MaxPool3d, Module, ModuleList, ReLU, BatchNorm3d, Tanh
from torchvision.transforms import CenterCrop
import torch.nn.functional as F
import torch

'''class Block3D(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the 3D convolution and ReLU layers
        self.conv1 = Conv3d(inChannels, outChannels, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv3d(outChannels, outChannels, kernel_size=3, padding=1)

    def forward(self, x):
        # apply CONV => RELU => CONV block
        return self.conv2(self.relu(self.conv1(x)))'''

class Block3D(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the 3D convolution, BatchNorm, and THAN layers
        self.conv1 = Conv3d(inChannels, outChannels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm3d(outChannels)
        self.than = ReLU()#Tanh()
        self.conv2 = Conv3d(outChannels, outChannels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm3d(outChannels)

    def forward(self, x):
        # apply CONV => BatchNorm => THAN => CONV => BatchNorm block
        x = self.bn2(self.conv2(self.than(self.bn1(self.conv1(x)))))
        return x

class Encoder3D(Module):
    def __init__(self, channels=(1, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block3D(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        # reduce the dimension
        self.pool = MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)

        return blockOutputs

class Decoder3D(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the upsampler blocks and decoder blocks
        self.upconvs = ModuleList(
            [ConvTranspose3d(channels[i], channels[i + 1], kernel_size=2, stride=2) for i in range(len(channels) - 1)]
        )
        self.dec_blocks = ModuleList(
            [Block3D(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, encFeatures, x):
        # Get the spatial size of the current tensor x
        (_, _, D, H, W) = x.shape
        # Use F.interpolate for resizing to match the required depth, height, and width
        encFeatures = F.interpolate(encFeatures, size=(D, H, W), mode='trilinear', align_corners=False)
        return encFeatures
    

class UNet3D(Module):
    def __init__(self, 
                 encChannels=(1, 16, 32, 64), 
                 decChannels=(64, 32, 16), 
                 nbClasses=1, 
                 retainDim=True, 
                 outSize=(60, 240, 240)):
        super().__init__()
        self.encoder = Encoder3D(encChannels)
        self.decoder = Decoder3D(decChannels)
        self.head = Conv3d(decChannels[-1], nbClasses, kernel_size=1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
        # pass the decoder features through the regression head
        map = self.head(decFeatures)
        
        # resize the output
        if self.retainDim:
            map = F.interpolate(map, size=self.outSize, mode="trilinear", align_corners=False)
        return map
