import torch.nn as nn
import numpy as np
import torch



def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups = 1):
    result = nn.Sequential()
    result.add_module("conv", nn.Conv3d(
                                    in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, groups=groups, bias=False))
    result.add_module("bn", nn.BatchNorm3d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, padding_mode = "zeros", deploy=False):


        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()


        if deploy:
            self.self.rbr_reparam = nn.Conv3d(in_channels=in_channels,
            out_channels=out_channels, kernel_size = kernel_size, stride=stride,
            padding=padding, dilation=dilation,groups=groups, bias= True, padding_mode=padding_mode)


        else:
            self.rbr_identity = nn.BatchNorm3d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)


    def forward(self, x):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    
    def get_equivalent_kernel_bias(self):

        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)

        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid


    def _pad_1x1_o_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

        
    def _fuse_bn_tensor(self, branch):

        if branch is None:
            return 0, 0
        
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_val = branch.bn.running_val
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps

        
        else:
            assert isinstance(branch,nn.BatchNorm3d)

            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups

                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1

                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_val  = branch.running_val 
            gamma = branch.weight
            beta  = branch.bias
            eps   = branch.eps


        std = (running_val + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma /std


    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False):

        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)

        self.cur_layer_idx= 1

        self.stage1 = self._make_stage(int(64  * width_multiplier[0]), num_blocks[0], stride=2)

        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)

        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)

        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2) 

        self.gap = nn.AdaptiveAvgPool3d(output_size = 1)

        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []

        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)

            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))

            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    
    def forward(self, x):
        out = self.stage0(x)    
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}



def create_RepVGG_A0(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A1(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A2(num_clases=1000, deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)

def create_RepVGG_B0(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1g2(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B1g4(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B2g2(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B2g4(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B3g2(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3g4(num_classes=1000, deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)



if __name__ == "__main__":
    model = create_RepVGG_B0(num_classes = 2)

    x = torch.randn((16, 3, 64, 224, 224))
    y = model(x)
    print(y)
    print(y.shape)
    # for n, m in model.named_modules():
    #     print(n, m)

