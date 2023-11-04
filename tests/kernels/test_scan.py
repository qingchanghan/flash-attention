import torch
from helper import TestDecorator

from kernels import cumsum


kt = TestDecorator()


@kt.case(dtypes=[torch.float], versions=["v0"], atol=1e-3, rtol=1e-3, ntest=1)
def test_scan(version="v0"):
    # rows = 16384
    # cols = 16384

    # rows = 8192
    # cols = 8192

    # rows = 1024
    # cols = 1024

    # rows = 1
    # cols = 1024

    # rows = 2
    # cols = 8192

    rows = 1024
    cols = 8192

    # rows = 2048
    # cols = 1024

    # rows = 128
    # cols = 128

    # rows = 64
    # cols = 32

    print(f"rows={rows}, cols={cols}")

    inp = kt.rand((rows, cols))

    def custom():
        output = cumsum(
            inp,
            version,
        )
        return output[0]

    def baseline():
        output = torch.cumsum(inp, dim=1).contiguous()
        return output

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0")
    kernel_list = [
        "test_scan",
    ]
    kt.run(kernel_list)

