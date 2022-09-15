import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()


        archive = zipfile.ZipFile('test_F_gelu.pnnx.bin', 'r')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        _, tmppath = tempfile.mkstemp()
        tmpf = open(tmppath, 'wb')
        with archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        tmpf.close()
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0, v_1, v_2):
        v_3 = F.gelu(input=v_2, approximate='none')
        v_4 = F.gelu(input=v_1, approximate='none')
        v_5 = F.gelu(input=v_0, approximate='none')
        v_6 = (v_5, v_4, v_3, )
        return v_6

def export_torchscript():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(16, dtype=torch.float)
    v_1 = torch.rand(2, 16, dtype=torch.float)
    v_2 = torch.rand(3, 12, 16, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1, v_2))
    mod.save("test_F_gelu_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(16, dtype=torch.float)
    v_1 = torch.rand(2, 16, dtype=torch.float)
    v_2 = torch.rand(3, 12, 16, dtype=torch.float)

    torch.onnx._export(net, (v_0, v_1, v_2), "test_F_gelu_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1', 'in2'], output_names=['out0'])

def test_inference():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(16, dtype=torch.float)
    v_1 = torch.rand(2, 16, dtype=torch.float)
    v_2 = torch.rand(3, 12, 16, dtype=torch.float)

    return net(v_0, v_1, v_2)
