import torch
import math
import utils
from utils import var_or_cuda
import numpy as np

class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.cube_len = args.cube_len

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.args.z_size, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, self.args.z_size, 1, 1, 1)
        #print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([100, 128, 16, 16, 16])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([100, 64, 32, 32, 32])
        out = self.layer5(out)
        #print(out.size())  # torch.Size([100, 1, 64, 64, 64])

        return out


class _D(torch.nn.Module):
    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args
        self.cube_len = args.cube_len

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU(self.args.leak_value)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=args.bias, padding=padd),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)
        #print(out.size()) # torch.Size([100, 1, 64, 64, 64])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([100, 64, 32, 32, 32])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([100, 128, 16, 16, 16])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer5(out)
        #print(out.size())  # torch.Size([100, 200, 1, 1, 1])

        return out


class _E(torch.nn.Module):

    def _get_padding(self,in_size,out_size, kernel_size, stride, dilation = 1):
        #padding = math.ceil(((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) / 2.0)
        padding = math.ceil((1/2)*(dilation*(kernel_size-1)+(out_size-1)*stride+(1-in_size)))
        return padding
    def _get_valid_padding(self,size,kernel_size,stride):
        padding = math.ceil(float(size - kernel_size + 1) / float(stride))
        return padding

    def output_features(self,size,kernel_size,stride,padding):

        out = (((size - kernel_size) + (2*padding)) // stride) + 1
        return out

    def __init__(self, args):
        super(_E, self).__init__()
        self.args = args
        self.img_size = args.image_size

        '''
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4,padding=self._get_padding(self.img_size,self.img_size,11,4)),
            torch.nn.BatchNorm2d( 64),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=self._get_padding(self.img_size,self.img_size,5,2) ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=self._get_padding(self.img_size,self.img_size,5,2) ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=self._get_padding(self.img_size,self.img_size,5,2)  ),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 400, kernel_size=8, stride=1, padding=self._get_valid_padding(self.img_size,5,2)  ),
            torch.nn.BatchNorm2d(400),
            torch.nn.ReLU()
        )
        '''
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 400, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(400),
            torch.nn.ReLU()
        )
        #get size of input to linear layers
        input = self.img_size
        for i in range(5):
            input = self.output_features(input,5,2,2)
        self.FC1 = torch.nn.Linear(400*input*input,200)
        self.FC2 = torch.nn.Linear(400*input*input, 200)

    def forward(self, x):
        out = x.view(self.args.batch_size, 3, self.img_size,self.img_size)# (Batch, Number Channels, height, width
        #print(out.size()) #torch.Size([32, 3, 224, 224])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([32, 64, 54, 54])
        out = self.layer2(out)
        #print(out.size())
        out = self.layer3(out)
        #print(out.size())
        out = self.layer4(out)
        #print(out.size())
        out = self.layer5(out)
        #print(out.size())

        out = out.view(self.args.batch_size,-1)#flatten
        #print(out.size())
        z_mean = self.FC1(out)
        z_log_var = self.FC2(out)
        #print(z_mean.size(),z_log_var.size())


        return z_mean, z_log_var

    def reparameterize(self, mu, var):
        if self.training:
            std = var.mul(0.5).exp_()
            eps = utils.var_or_cuda((std.data.new(std.size()).normal_()))
            z =  eps.mul(std).add_(mu)
            return z
        else:
            return mu

class _E_MultiView(torch.nn.Module):

    def _get_padding(self,in_size,out_size, kernel_size, stride, dilation = 1):
        #padding = math.ceil(((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) / 2.0)
        padding = math.ceil((1/2)*(dilation*(kernel_size-1)+(out_size-1)*stride+(1-in_size)))
        return padding
    def _get_valid_padding(self,size,kernel_size,stride):
        padding = math.ceil(float(size - kernel_size + 1) / float(stride))
        return padding

    def output_features(self,size,kernel_size,stride,padding):

        out = (((size - kernel_size) + (2*padding)) // stride) + 1
        return out

    def __init__(self, args):
        super(_E_MultiView, self).__init__()
        self.args = args
        self.img_size = args.image_size
        self.combine_type = args.combine_type

        '''
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4,padding=self._get_padding(self.img_size,self.img_size,11,4)),
            torch.nn.BatchNorm2d( 64),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=self._get_padding(self.img_size,self.img_size,5,2) ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=self._get_padding(self.img_size,self.img_size,5,2) ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=self._get_padding(self.img_size,self.img_size,5,2)  ),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 400, kernel_size=8, stride=1, padding=self._get_valid_padding(self.img_size,5,2)  ),
            torch.nn.BatchNorm2d(400),
            torch.nn.ReLU()
        )
        '''
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 400, kernel_size=5, stride=2,padding=2),
            torch.nn.BatchNorm2d(400),
            torch.nn.ReLU()
        )
        #get size of input to linear layers
        input = self.img_size
        for i in range(5):
            input = self.output_features(input,5,2,2)
        self.FC1 = torch.nn.Linear(400*input*input,200)
        self.FC2 = torch.nn.Linear(400*input*input, 200)

    def forward(self,images):
        means = utils.var_or_cuda( torch.zeros(self.args.num_views,self.args.batch_size,200) )
        vars = utils.var_or_cuda( torch.zeros(self.args.num_views,self.args.batch_size,200) )
        zs = utils.var_or_cuda( torch.zeros(self.args.num_views,self.args.batch_size,200) )
        for i, image in enumerate(images):
            image = utils.var_or_cuda(image)
            z_mean, z_log_var = self.single_image_forward(image)
            zs[i:] = self.reparameterize(z_mean,z_log_var)
            means[i:] = z_mean
            vars[i:] = z_log_var

        #z_mu= self.combine(means)
        #z_var = self.combine(vars)
        return self.combine(zs), means, vars

    def combine(self,input):
        #input should be (num_views,batchsize,z_dimension)
        #output should be (batch_size,z_dimension)
        if self.combine_type == 'mean':
            output =  torch.mean(input,0)
        elif self.combine_type == 'max':
            output = torch.max(input, 0)[0]
        elif self.combine_type == 'concat':
            pass
        return output
    def single_image_forward(self, x):
        out = x.view(self.args.batch_size, 3, self.img_size,self.img_size)# (Batch, Number Channels, height, width
        #print(out.size()) #torch.Size([32, 3, 224, 224])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([32, 64, 54, 54])
        out = self.layer2(out)
        #print(out.size())
        out = self.layer3(out)
        #print(out.size())
        out = self.layer4(out)
        #print(out.size())
        out = self.layer5(out)
        #print(out.size())

        out = out.view(self.args.batch_size,-1)#flatten
        #print(out.size())
        z_mean = self.FC1(out)
        z_log_var = self.FC2(out)
        #print(z_mean.size(),z_log_var.size())


        return z_mean, z_log_var

    def reparameterize(self, mu, var):
        if self.training:
            std = var.mul(0.5).exp_()
            eps = utils.var_or_cuda((std.data.new(std.size()).normal_()))
            z =  eps.mul(std).add_(mu)
            return z
        else:
            return mu