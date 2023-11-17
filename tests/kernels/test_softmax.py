import torch
import oneflow as flow
from helper import TestDecorator

from kernels import softmax


kt = TestDecorator()


# @kt.case(dtypes=[torch.half], versions=["v0", "v1"], atol=1e-3, rtol=1e-3, ntest=1)
# @kt.case(dtypes=[torch.half], versions=["v1"], atol=1e-3, rtol=1e-3, ntest=1)
# @kt.case(dtypes=[torch.half], versions=["v1", "v2"], atol=1e-3, rtol=1e-3, ntest=1)
@kt.case(dtypes=[torch.float], versions=["v0", "v1", "v2"], atol=1e-3, rtol=1e-3, ntest=1)
def test_softmax(version="v0"):
    # rows = 16384
    # cols = 16384

    rows = 8192
    cols = 8192

    # rows = 1024
    # cols = 1024

    # rows = 1
    # cols = 256

    # rows = 2048
    # cols = 1024

    # rows = 128
    # cols = 128

    # rows = 64
    # cols = 32

    print(f"rows={rows}, cols={cols}")

    inp = kt.rand((rows, cols))
    # inp_np = inp.detach().cpu().numpy()
    # inp_of = flow.tensor(inp_np).cuda()

    def custom():
        output = softmax(
            inp,
            version,
        )
        return output[0]

    def baseline():
        output = torch.nn.functional.softmax(inp, dim=-1).contiguous()

        # output_of = flow.nn.functional.softmax(inp_of, dim=-1)
        # output_np = output_of.numpy()
        # output = torch.from_numpy(output_np).cuda()

        return output

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0")
    kernel_list = [
        "test_softmax",
    ]
    kt.run(kernel_list)

