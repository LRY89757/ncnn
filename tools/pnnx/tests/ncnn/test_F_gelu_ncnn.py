import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(16, dtype=torch.float)
    in1 = torch.rand(2, 16, dtype=torch.float)
    in2 = torch.rand(3, 12, 16, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
         net.load_param("test_F_gelu.ncnn.param")
         net.load_model("test_F_gelu.ncnn.bin")

         with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.numpy()).clone())
            ex.input("in1", ncnn.Mat(in1.numpy()).clone())
            ex.input("in2", ncnn.Mat(in2.numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)))
            _, out1 = ex.extract("out1")
            out.append(torch.from_numpy(np.array(out1)))
            _, out2 = ex.extract("out2")
            out.append(torch.from_numpy(np.array(out2)))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)
