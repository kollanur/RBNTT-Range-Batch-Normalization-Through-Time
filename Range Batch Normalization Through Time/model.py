import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys



def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


def batch_norm(x, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        
        x_hat = ((x - moving_mean) / torch.sqrt(moving_var + eps))
    
    else:
        assert len(x.shape) in (2, 4)
        if(len(x.shape) == 2):
            
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim = 0)
            
        else:
            
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            
        x_hat = (x - mean) / torch.sqrt(var + eps)
        
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    
    y = gamma * x_hat + beta    # scale and shift
    
    return y, moving_mean.data, moving_var.data
    
    
def range_batch_norm(x, gamma, beta, moving_mean, moving_var, eps, momentum):
    
    batch_size = x.shape[0]
    c =  ( 0.5 * 0.35 ) * ( 1 + (math.pi * math.log(4)) ** 0.5) / ((2 * math.log(batch_size)) ** 0.5)
    
    if not torch.is_grad_enabled():
        
        x_hat = (x - moving_mean) / (moving_var)
        
    else:
        assert len(x.shape) in (2,4)
        if(len(x.shape) == 2):
            
            mean = x.mean(dim = 0)
            rang = (x - mean)
            var = (torch.max(rang,0).values - torch.min(rang,0).values)
            
        else:
            num_features = x.shape[1]
    
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            rang = (x - mean)
            var = (torch.max((rang.reshape(num_features, -1)),1).values.reshape(1 ,num_features ,1 ,1) - torch.min((rang.reshape(num_features, -1)),1).values.reshape(1 ,num_features ,1 ,1))
        
        scale = c * var
        x_hat = (x - mean) / scale
        
        
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * scale
        
    #y = gamma * x_hat + beta   # scale and shfit
    y = gamma * x_hat
    
    return y, moving_mean.data, moving_var.data
    
class RangeBatchNorm2d(nn.Module):
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True ):
        super(RangeBatchNorm2d, self).__init__()
        
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        
        
    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        
        y, self.moving_mean, self.moving_var = range_batch_norm(x, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return y
    
class RangeBatchNorm1d(nn.Module):
     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True ):
        super(RangeBatchNorm1d, self).__init__()
        
        shape = (1, num_features)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        
    
     def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        
        y, self.moving_mean, self.moving_var = range_batch_norm(x, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return y

class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


class SNN_VGG9_RBNTT(nn.Module):
    
    def __init__(self, num_steps, leak_mem=0.95, img_size=32, num_cls=10):
        super(SNN_VGG9_RBNTT, self).__init__()
        
        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        
        print(">>>>>>>>>>>>>>>>>>>> VGG9 RBN >>>>>>>>>>>>>>>>>>>>")
        print("************** time step per batchnorm".format(self.batch_num))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        affine_flag = True
        bias_flag = True
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias = bias_flag)
        self.rbntt1 = nn.ModuleList([RangeBatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias = bias_flag)
        self.rbntt2 = nn.ModuleList([RangeBatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias = bias_flag)
        self.rbntt3 = nn.ModuleList([RangeBatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias = bias_flag)
        self.rbntt4 = nn.ModuleList([RangeBatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias = bias_flag)
        self.rbntt5 = nn.ModuleList([RangeBatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias = bias_flag)
        self.rbntt6 = nn.ModuleList([RangeBatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias = bias_flag)
        self.rbntt7 = nn.ModuleList([RangeBatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear((self.img_size//8)*(self.img_size//8)*256, 1024, bias=bias_flag)
        self.rbntt_fc = nn.ModuleList([RangeBatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)
        
        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.rbntt_list = [self.rbntt1, self.rbntt2, self.rbntt3, self.rbntt4, self.rbntt5, self.rbntt6, self.rbntt7, self.rbntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]
        
        
        # Turn off bias of BNTT
        for rbn_list in self.rbntt_list:
            for rbn_temp in rbn_list:
                rbn_temp.bias = None
                
        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
                
    def forward(self, inp):
        
        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]
        
        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()
        
        
        for t in range(self.num_steps):
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp
            
            
            for i in range(len(self.conv_list)):
                
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.rbntt_list[i][t](self.conv_list[i](out_prev))
                
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                
                
                out = self.spike_fn(mem_thr)
                
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                
                mem_conv_list[i] = mem_conv_list[i] - rst
                
                out_prev = out.clone()
                
                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()
                    
            out_prev = out_prev.reshape(batch_size, -1)
            
            
            mem_fc1 = self.leak_mem * mem_fc1 + self.rbntt_fc[t](self.fc1(out_prev))
            
            
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()
            
            # accumulate voltagfe in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)
            
            
        out_voltage = mem_fc2 / self.num_steps
        
        return out_voltage


class SNN_VGG11_RBNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32,  num_cls=10):
        super(SNN_VGG11_RBNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps

        print (">>>>>>>>>>>>>>>>> VGG11 RBN>>>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False




        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.rbntt1 = nn.ModuleList([nn.RangeBatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.rbntt2 = nn.ModuleList([nn.RangeBatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.rbntt3 = nn.ModuleList([nn.RangeBatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.rbntt4 = nn.ModuleList([nn.RangeBatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.rbntt5 = nn.ModuleList([nn.RangeBatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.rbntt6 = nn.ModuleList([nn.RangeBatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.rbntt7 = nn.ModuleList([nn.RangeBatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.rbntt8 = nn.ModuleList([nn.RangeBatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AdaptiveAvgPool2d((1,1))


        self.fc1 = nn.Linear(512, 4096, bias=bias_flag)
        self.rbntt_fc = nn.ModuleList([nn.RangeBatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(4096, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]
        self.rbntt_list = [self.rbntt1, self.rbntt2, self.rbntt3, self.rbntt4, self.rbntt5, self.rbntt6, self.rbntt7, self.rbntt8, self.rbntt_fc]
        self.pool_list = [self.pool1, self.pool2, False, self.pool3, False, self.pool4, False, self.pool5]

        # Turn off bias of BNTT
        for bn_list in self.rbntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv3 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv5 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
        mem_conv6 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
        mem_conv7 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv8 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7, mem_conv8]

        mem_fc1 = torch.zeros(batch_size, 4096).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.num_steps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.rbntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.rbntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)


        out_voltage = mem_fc2 / self.num_steps

        return out_voltage
