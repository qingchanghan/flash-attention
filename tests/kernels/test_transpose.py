import torch
from helper import TestDecorator

from kernels import matrix_transpose


kt = TestDecorator()


# @kt.case(versions=["v0", "v1", "v2"], atol=1e-3, rtol=1e-3, ntest=1)
# @kt.case(dtypes=[torch.half], versions=["v0", "v1", "v2", "v3"], atol=1e-3, rtol=1e-3, ntest=1)
# @kt.case(dtypes=[torch.half], versions=["v3", "v4"], atol=1e-3, rtol=1e-3, ntest=1)
# @kt.case(dtypes=[torch.half], versions=["v0", "v1", "v2", "v3", "v4"], atol=1e-3, rtol=1e-3, ntest=1, nrepeat=1) # for ncu
# @kt.case(dtypes=[torch.half], versions=["v0", "v1", "v2", "v3", "v4", "v5"], atol=1e-3, rtol=1e-3, ntest=1)
@kt.case(dtypes=[torch.half], versions=["v4", "v5", "v6"], atol=1e-3, rtol=1e-3, ntest=1)
# @kt.case(dtypes=[torch.half], versions=["v4", "v5"], atol=1e-3, rtol=1e-3, ntest=1, nrepeat=1)
def test_transpose(version="v0"):
    # rows = 16384
    # cols = 16384

    # rows = 8192
    # cols = 8192

    rows = 1024
    cols = 1024

    # rows = 2048
    # cols = 1024

    # rows = 128
    # cols = 128

    # rows = 64
    # cols = 32

    print(f"rows={rows}, cols={cols}")

    inp = kt.rand((rows, cols))

    def custom():
        output = matrix_transpose(
            inp,
            version,
        )
        return output[0]

    def baseline():
        output = torch.transpose(inp, 0, 1).contiguous()
        return output

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0")
    kernel_list = [
        "test_transpose",
    ]
    kt.run(kernel_list)

