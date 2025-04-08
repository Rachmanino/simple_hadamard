import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout
 
import math
import torch

def is_pow_of_2(n):
    """Check if n is a power of 2."""
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0

def hadamard(b, n, dtype):
    """
    A block is responsible for computing one row.
    The configuration is based on https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/csrc/fast_hadamard_transform_cuda.cu
    """
    assert is_pow_of_2(n), "n must be a power of 2"
    assert 2 <= n <= 32768, "n must be in [2, 32768]"
    assert dtype in ['float32'], "dtype must be float32"    
    elem_size = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2
    }[dtype]

    logN = int(math.log2(n))
    threads = [0, 1, 1, 1, 2, 4, 8, 16, 32, 32, 128, 256, 256, 256, 256, 256][logN]
    thread_elem = n // threads # Each thread is responsible for a chunk of elements
    thread_round = int(math.log2(thread_elem))

    warps = 1 if threads <= 32 else threads // 32 
    warp_round = int(math.log2(threads/warps))

    block_round = int(math.log2(warps))

    exchange_round = n * elem_size // 32768 if n * elem_size > 32768 else 1
    thread_elem_in_smem = thread_elem // exchange_round if exchange_round > 1 else thread_elem

    # debug log
    # print(f'{threads=}, {thread_round=}')
    # print(f'{warps=}, {warp_round=}')
    # print(f'{block_round=}')
    
    @T.prim_func
    def main(
        A: T.Buffer((b, n), dtype), # Should be T.Tensor(..) here for TileLang new versions
        B: T.Buffer((b, n), dtype),
    ): 
        with T.Kernel(b, threads=threads) as bx:
            local = T.alloc_local((thread_elem,), dtype)
            shared = T.alloc_shared((threads, thread_elem_in_smem), dtype)
            T.annotate_layout({
                shared: make_mma_swizzle_layout(shared)
            })
            # T.use_swizzle(panel_size=10, enable=True)

            tx = T.get_thread_binding(0)

            # 1. Load from HBM to register
            for i in T.vectorized(thread_elem):
                local[i] = A[bx, tx*thread_elem + i]

            # 2. Hadamard inside thread, n<=8
            for i in T.serial(thread_round):
                chunksize = 1 << (i + 1)
                chunknum = thread_elem // chunksize
                for j in T.serial(chunknum):
                    chunkbase = j * chunksize
                    for k in T.serial(chunksize // 2):
                        local[chunkbase + k] = local[chunkbase + k] + local[chunkbase + k + chunksize // 2]
                        local[chunkbase + k + chunksize // 2] = local[chunkbase + k] - 2 * local[chunkbase + k + chunksize // 2]

            for cur_round in T.Pipelined(exchange_round, num_stages=3):
                exchange_base = thread_elem_in_smem * cur_round

                # 3. Hadamard inside warp, n<=512
                # In warp level, we can directly compute on shared memory, since warps are scheduled simultaneously
                for i in T.vectorized(thread_elem_in_smem):
                    shared[tx, i] = local[exchange_base + i]
            
                for i in T.serial(warp_round):
                    tx_stride = 1 << i
                    sign = (tx >> i) & 1 # get i-th lowest bit of tx, which determines the operation type for shared[tx, :]
                    for j in T.vectorized(thread_elem_in_smem):
                        shared[tx, j] = T.if_then_else(
                            sign == 0,
                            shared[tx, j] + shared[tx ^ tx_stride, j],
                            shared[tx ^ tx_stride, j] - shared[tx, j]
                        )
                    
                for i in T.vectorized(thread_elem_in_smem):
                    local[exchange_base+i] = shared[tx, i]
            
                # 4. Hadamard inside block, n<=32768
                # Only exchange once for n<=8192, since shared mem can hold all elems
                for i in T.serial(block_round):
                    tx_stride = 1 << (warp_round + i)
                    sign = (tx >> (warp_round + i)) & 1 # get i-th lowest bit of tx, which determines the operation type for shared[tx, :]
                    for j in T.vectorized(thread_elem_in_smem):
                        local[exchange_base+j] = T.if_then_else(
                            sign == 0,
                            local[exchange_base+j] + shared[tx ^ tx_stride, j],
                            shared[tx ^ tx_stride, j] - local[exchange_base+j]
                        )
                    
                    for j in T.vectorized(thread_elem_in_smem):
                        shared[tx, j] = local[exchange_base+j]
            
            # 5. Write back to HBM
            for i in T.vectorized(thread_elem):
                B[bx, tx*thread_elem + i] = local[i]
            
    return main




           