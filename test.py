import unittest
from ClassifierBlock import *
from ConvolutionBlock import *
from DDModule import *
from DeformableConvolutionBlock import *
from FeatureBlock import *
from SELayer import *
from SeparableConv2D import *
from TDSNet import *

class MyTestCase(unittest.TestCase):
    def test_FeatureBolck(self):
        batch_size = 32
        in_channels = 1
        h,w = 256,256
        out_channels = 64
        test = FeatureBlock(out_channels)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,out_channels,h//4,w//4])
        )

    def test_DDPathBolckSize1(self):
        batch_size = 32
        in_channels = 1
        h,w = 256,256
        out_channels = 64
        kernel_size = 3
        dilation = 2
        test = DDPath(out_channels,kernel_size,dilation)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,out_channels,h,w])
        )

    def test_DDPathBolckSize2(self):
        batch_size = 32
        in_channels = 1
        h,w = 256,256
        out_channels = 64
        kernel_size = 5
        dilation = 3
        test = DDPath(out_channels,kernel_size,dilation)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,out_channels,h,w])
        )

    def test_DeformableConvolutionBlock(self):
        batch_size = 32
        in_channels = 1
        h,w = 256,256
        out_channels = 64
        test = DeformableConvolutionBlock(in_channels,out_channels)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,out_channels,h,w])
        )

    def test_DDModel(self):
        batch_size = 32
        in_channels = 1
        h,w = 256,256
        out_channels = 64
        test = DDModel(out_channels)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,out_channels*7,h,w])
        )

    def test_ConvolutionBlock1(self):
        batch_size = 32
        in_channels = 1
        h,w = 256,256
        out_channels = 64
        conv_num = 3
        test = ConvolutionBlock(conv_num,out_channels)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,out_channels,h//2,w//2])
        )

    def test_ConvolutionBlock2(self):
        batch_size = 32
        in_channels = 1
        h,w = 256,256
        out_channels = 64
        test = ConvolutionBlock(1,out_channels,first=True)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,out_channels,h,w])
        )

    def test_SeparableConv2D(self):
        batch_size = 32
        in_channels = 128
        h,w = 256,256
        test = SeparableConv2D(128)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,1024,h,w])
        )

    def test_SELayer(self):
        batch_size = 32
        in_channels = 128
        h,w = 256,256
        test = SELayer(in_channels)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,in_channels,h,w])
        )

    def test_ClassifierBlock(self):
        batch_size = 32
        in_channels = 128
        num_class = 6
        h,w = 256,256
        test = ClassifierBlock(num_class,in_channels)
        self.assertEqual(
            test(torch.rand(batch_size,in_channels,h,w)).shape,
            torch.Size([batch_size,num_class])
        )


if __name__ == '__main__':
    unittest.main()
