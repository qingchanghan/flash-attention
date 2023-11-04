import random
import time
from collections import OrderedDict
from functools import wraps

import numpy as np
import torch


def cast_fp32_tensor(tlist):
    return [ele.to(torch.float32) for ele in tlist]


def is_nan(x):
    return x.isnan().any().item()


def is_inf(x):
    return x.isinf().any().item()


use_ncu = False
# use_ncu = True


class TestDecorator(object):
    def __init__(self):
        self.all_case = OrderedDict()
        self.dtypes = [torch.float, torch.half, torch.bfloat16]
        self.dtype = None

    def init(self, device):
        # device: str. e.g. "cuda:0"
        self.device = torch.device(device)

    @property
    def io_factor(self):
        if self.dtype == torch.float32:
            return 4
        else:
            return 8

    def move(self, data):
        return data.to(self.device, dtype=self.dtype)

    def norm_res_list(self, rlist):
        return [ele.to(dtype=self.dtype).contiguous() for ele in rlist]

    def rand(self, shape):
        return self.move(torch.rand(shape))

    def randint(self, low, high, shape):
        return torch.randint(low, high, shape).to(self.device, dtype=torch.long)

    def ones(self, shape):
        return self.move(torch.ones(shape))

    def zeros(self, shape):
        return self.move(torch.zeros(shape))

    def attn_mask(self, batch_size, seq_len, dtype=None):
        """
        1 for padding tokens , 0 for non-padding tokens
        """
        if dtype is None:
            dtype = self.dtype
        mask = torch.zeros((batch_size, seq_len))
        for b in range(batch_size):
            valid_seq_len = random.randint(1, seq_len)
            mask[b, valid_seq_len:] = 1
        return mask.to(self.device, dtype=dtype)

    def dec_self_attn_mask(self, seq_len, dtype=None):
        """
        e.g. if seq_len = 3
        return:
        0 1 1
        0 0 1
        0 0 0
        """
        if dtype is None:
            dtype = self.dtype
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.to(self.device, dtype=dtype)

    def case(self, dtypes=list(), versions=list(), ntest=5, nrepeat=20, rtol=1e-5, atol=1e-5):
        if not dtypes:
            dtypes = self.dtypes
        if not versions:
            versions = ["v0"]

        def decorator(func):
            self.all_case[func.__name__] = (
                func, dtypes, versions, ntest, nrepeat, rtol, atol)
            # @wraps(func)
            # def wrapper(*args, **kwargs):
            #     return func(*args, **kwargs)
            # return wrapper
            return func

        return decorator

    def assert_allclose(self, tlist1, tlist2, rtol, atol):
        """
        tlist1 and tlist2 are list of torch.tensor.
        """
        if use_ncu:
            return
        passed = True
        # tlist1 = [ele.cpu().numpy().flatten() for ele in tlist1]
        # tlist2 = [ele.cpu().numpy().flatten() for ele in tlist2]
        assert len(tlist1) == len(tlist2)
        for i in range(len(tlist1)):
            t1 = tlist1[i]
            t2 = tlist2[i]
            if t1 is None and t2 is None:
                continue
            # fast allclose
            res = torch.allclose(
                t1.flatten(), t2.flatten(), rtol=rtol, atol=atol, equal_nan=False
            )
            if res:
                continue
            passed = False
            print("torch.allclose failed, use numpy.allclose to log detail.")
            t1 = t1.cpu().numpy().flatten()
            t2 = t2.cpu().numpy().flatten()
            try:
                np.testing.assert_allclose(
                    t1, t2, rtol=rtol, atol=atol, verbose=True)
            except Exception as ex:
                print(f"Unmatches in the {i}-th tensor.")
                print(ex)
                continue
        if not passed:
            exit(0)

    def test(self, custom, baseline, nrepeat, rtol, atol):
        """
        (custom() − baseline()) <= atol + rtol * abs(baseline)
        """

        def core(func):
            res = func()  # warmup for GPU
            self.assert_allclose(res, res, rtol, atol)
            timing = list()
            for i in range(nrepeat):
                torch.cuda.synchronize(device=self.device)
                begin = time.time()
                cur_res = func()
                torch.cuda.synchronize(device=self.device)
                # In seconds
                timing.append(time.time() - begin)
                self.assert_allclose(cur_res, res, rtol, atol)
            return res, np.mean(timing) * 1000

        # print("Run baseline...")
        baseline_res, baseline_time = core(baseline)

        # import pdb; pdb.set_trace()

        # print("Run custom...")
        custom_res, custom_time = core(custom)

        if not use_ncu:
            # print("Compare the results of custom and baseline...")
            self.assert_allclose(custom_res, baseline_res, rtol, atol)
            print(
                "Test passed. Time of custom/baseline (ms): %.3f / %.3f, speedup: %.3f"
                % (custom_time, baseline_time, baseline_time / custom_time)
            )

    def run(self, case_names):
        if case_names is None:
            case_names = self.all_case.keys()
        for cn in case_names:
            assert cn in self.all_case, "Illegal case name to be tested."
            func, dtypes, versions, ntest, nrepeat, rtol, atol = self.all_case[cn]
            for i in range(ntest):
                for version in versions:
                    for dtype in dtypes:
                        self.dtype = dtype
                        print(
                                ">>>>>>>>>>>>>>>>>>>>>>" f"{cn}, ntest [{i}], version: [{version}], dtype [{dtype}]:"
                        )
                        custom, baseline = func(version=version)
                        torch.cuda.synchronize(device=self.device)
                        self.test(custom, baseline, nrepeat, rtol, atol)


def flat_dim(idxs, dims):
    assert len(idxs) == len(dims) or len(idxs) == len(dims) + 1
    base = 1
    res = 0
    dims = dims[::-1]
    idxs = idxs[::-1]
    if len(idxs) == len(dims) + 1:
        dims.append(0)
        for idx, dim in zip(idxs, dims):
            assert idx < dim
            res += idx * base
            base *= dim
    return res


def expand_dim(idx, dims):
    res = [0] * len(dims)
    for i, d in enumerate(dims[::-1]):
        res[i] = idx % d
        idx //= d
        if idx == 0:
            break
        assert idx == 0
    return res[::-1]


