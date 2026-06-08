"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

  dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
  sparse_flash_forward   (A2) block-sparse flash attention forward
  sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import torch
import triton
import triton.language as tl

# A1 kernel
@triton.jit
def _dsd_kernel(
    Av_ptr, # pointer for values (nnz, block, block) f32, contiguous
    Aro_ptr, # rows offsets ptr (n_block_rows + 1,), i32
    Aci_ptr, # column indices ptr (nnz,), i32
    B_ptr, C_ptr, # (K, N) and (M, N) f32
    N,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK: tl.constexpr, # = block (square)
    BLOCK_N: tl.constexpr, # N tile
    INNER: tl.constexpr, # K-dim chunk size (splits BLOCK to stay within shared-mem limit)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # [row_start, row_end) are the live blocks for this block-row
    row_start = tl.load(Aro_ptr + pid_m)
    row_end = tl.load(Aro_ptr + pid_m + 1)

    offs_blk = tl.arange(0, BLOCK) # within-block index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # N cols for this tile
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK, BLOCK_N), dtype=tl.float32) # accumulator (with fp32)

    for idx in range(row_start, row_end): # only live blocks!!!!
        kcol = tl.load(Aci_ptr + idx) # the K-block of this live block (corresponding to B)

        # Split the BLOCK-wide inner K dimension into INNER-sized chunks so that
        # each tl.dot uses at most (BLOCK*INNER + INNER*BLOCK_N) fp32 of shared mem.
        for k in range(BLOCK // INNER):
            r_inner = k * INNER + tl.arange(0, INNER)
            # A chunk: (BLOCK, INNER)
            # columns [k*INNER : (k+1)*INNER] of this A block
            Av = tl.load(Av_ptr + idx * BLOCK * BLOCK + offs_blk[:, None] * BLOCK + r_inner[None, :])
            # B chunk: (INNER, BLOCK_N)
            # rows matching those A columns
            Bv = tl.load(
                B_ptr + (kcol * BLOCK + r_inner[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                mask=n_mask[None, :], other=0.0,
            )
            acc += tl.dot(Av, Bv, allow_tf32=False)

    offs_cm = pid_m * BLOCK + offs_blk
    tl.store(
        C_ptr + offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=n_mask[None, :],
    )

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    qro_ptr, qci_ptr,
    sm_scale,
    T,
    stride_b, stride_t, stride_d,
    stride_lb,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    bh = tl.program_id(0) # (batch, head)
    qb = tl.program_id(1) # query block

    row_start = tl.load(qro_ptr + qb)
    row_end = tl.load(qro_ptr + qb + 1)

    offs_q = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_q < T
    d_mask = offs_d < D
    # (BLOCK_Q, BLOCK_D) tile of Q for this block and head
    q = tl.load(Q_ptr + bh * stride_b + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    # online softmax accumulators in log2 space
    # O(T^2) to O(T) memory trick
    # for each query block, keep track of the max score and sum of exp scores
    m_i = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32) # max score for each query
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32) # sum of exp scores for each query
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    # predefience outside the loop
    rk = tl.arange(0, BLOCK_K)
    # exp(x) = exp2(x * log2(e))
    qk_scale = sm_scale * LOG2E

    for idx in range(row_start, row_end):
        kb = tl.load(qci_ptr + idx) # key block
        offs_k = kb * BLOCK_K + rk
        mask_k  = offs_k < T

        k = tl.load(K_ptr + bh * stride_b + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                    mask=mask_k[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(V_ptr + bh * stride_b + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                    mask=mask_k[:, None] & d_mask[None, :], other=0.0)

        # scores in log2 units (for stable softmax later)
        # q (BLOCK_Q, d) @ k^T (d, BLOCK_K) -> s (BLOCK_Q, BLOCK_K)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale # base-2 log
        s = tl.where(mask_k[None, :], s, float('-inf'))

        # flash online softmax (O(T^2) to O(T) memory trick)
        m_new  = tl.maximum(m_i, tl.max(s, axis=1))
        exp_s  = tl.exp2(s - m_new[:, None]) # exp scores for this block
        alpha  = tl.exp2(m_i - m_new) # rescaling factor for the old acc vals

        l_i = alpha * l_i + tl.sum(exp_s, axis=1)
        acc = acc * alpha[:, None] + tl.dot(exp_s.to(q.dtype), v).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    # L = LOG2E * logsumexp(sigma QK^T), same as m_i + log2(l_i)
    # since exp2(m_i) * l_i is denominator
    L_val = m_i + tl.log2(l_i)

    tl.store(O_ptr + bh * stride_b + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
             acc.to(tl.float16), mask=q_mask[:, None] & d_mask[None, :])
    
    tl.store(L_ptr + bh * stride_lb + offs_q, L_val, mask=q_mask)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    qro_ptr, qci_ptr,
    sm_scale, T,
    stride_b, stride_t, stride_d, stride_lb,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    bh = tl.program_id(0)
    qb = tl.program_id(1)
    row_start = tl.load(qro_ptr + qb)
    row_end = tl.load(qro_ptr + qb + 1)

    offs_q = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_q < T
    d_mask = offs_d < D
    qd_mask = q_mask[:, None] & d_mask[None, :]
    base = bh * stride_b

    # Load Q_i and dO_i once per kernel instance
    q  = tl.load(Q_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                 mask=qd_mask, other=0.0)
    do = tl.load(dO_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                 mask=qd_mask, other=0.0)

    L_i = tl.load(L_ptr + bh * stride_lb + offs_q, mask=q_mask, other=0.0)
    D_i = tl.load(D_ptr + bh * stride_lb + offs_q, mask=q_mask, other=0.0)

    qk_scale = sm_scale * LOG2E
    acc_dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for idx in range(row_start, row_end):
        kb  = tl.load(qci_ptr + idx)
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < T
        kd_mask = k_mask[:, None] & d_mask[None, :]

        k = tl.load(K_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                    mask=kd_mask, other=0.0)
        v = tl.load(V_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                    mask=kd_mask, other=0.0)

        s  = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        s  = tl.where(k_mask[None, :], s, float('-inf'))

        p  = tl.exp2(s - L_i[:, None])

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - D_i[:, None])

        acc_dq += tl.dot(ds.to(q.dtype), k).to(tl.float32) * sm_scale

    tl.store(dQ_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
             acc_dq.to(tl.float16), mask=qd_mask)


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    L_ptr, D_ptr, dK_ptr, dV_ptr,
    kro_ptr, kci_ptr,
    sm_scale,
    T,
    stride_b, stride_t, stride_d, stride_lb,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    bh = tl.program_id(0)
    kb = tl.program_id(1)

    row_start = tl.load(kro_ptr + kb)
    row_end = tl.load(kro_ptr + kb + 1)

    offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    k_mask = offs_k < T
    d_mask = offs_d < D
    kd_mask = k_mask[:, None] & d_mask[None, :]
    base = bh * stride_b

    # Load K_j, V_j once per kernel instance
    k = tl.load(K_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                mask=kd_mask, other=0.0)
    v = tl.load(V_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
                mask=kd_mask, other=0.0)

    acc_dk = tl.zeros((BLOCK_K, D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_K, D), dtype=tl.float32)
    qk_scale = sm_scale * LOG2E

    for idx in range(row_start, row_end):
        qb  = tl.load(kci_ptr + idx)
        offs_q = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < T
        qd_mask = q_mask[:, None] & d_mask[None, :]

        q  = tl.load(Q_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                     mask=qd_mask, other=0.0)
        do = tl.load(dO_ptr + base + offs_q[:, None] * stride_t + offs_d[None, :] * stride_d,
                     mask=qd_mask, other=0.0)

        L_i = tl.load(L_ptr + bh * stride_lb + offs_q, mask=q_mask, other=0.0)
        D_i = tl.load(D_ptr + bh * stride_lb + offs_q, mask=q_mask, other=0.0)

        s  = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        s  = tl.where(q_mask[:, None] & k_mask[None, :], s, float('-inf'))

        P  = tl.exp2(s - L_i[:, None])

        acc_dv += tl.dot(tl.trans(P).to(do.dtype), do).to(tl.float32)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = P * (dp - D_i[:, None])

        acc_dk += tl.dot(tl.trans(ds).to(q.dtype), q).to(tl.float32) * sm_scale

    tl.store(dK_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
             acc_dk.to(tl.float16), mask=kd_mask)
    tl.store(dV_ptr + base + offs_k[:, None] * stride_t + offs_d[None, :] * stride_d,
             acc_dv.to(tl.float16), mask=kd_mask)


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.

    Inputs:
      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32   dense right operand
      M, K, N, block                      ints   dims and block size
    Returns:
      C              (M, N)               fp32

    fp32 throughout, allow_tf32=False.

    TODO: implement.
    """
    # this function is a wrapper (hosted by CPU)
    C = torch.zeros(M, N, device=B.device, dtype=torch.float32) # output allocation
    BLOCK_N = 128
    INNER = 64
    shm_per_stage = (block * INNER + INNER * BLOCK_N) * 4 # bytes
    num_stages = 1 if shm_per_stage * 2 > 163_000 else 2

    grid = (M // block, triton.cdiv(N, BLOCK_N)) # worker grid (M/block, N/BLOCK_N)
    _dsd_kernel[grid](
        values, row_offsets, column_indices, B, C,
        N,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK=block, BLOCK_N=BLOCK_N, INNER=INNER,
        num_stages=num_stages,
    )
    return C


def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.

    Inputs:
      Q, K, V        (B, H, T, d)         fp16
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query
      q_col_indices  (nnz,)               int32  block i, its live key blocks j
      sm_scale       float                       1/sqrt(d)
      BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
      O              (B, H, T, d)         fp16
      L              (B, H, T)            fp32   log2 of the softmax denominator (sec 3)

    See ALGORITHMS.md sec 3 for O and L.

    TODO: implement.
    """
    B, H, T, d = Q.shape
    BH = B * H
    Qf, Kf, Vf = Q.reshape(BH, T, d), K.reshape(BH, T, d), V.reshape(BH, T, d) # flatten B and H
    O = torch.empty_like(Q)
    Of = O.reshape(BH, T, d)
    L = torch.empty(B, H, T, device=Q.device, dtype=torch.float32)
    Lf = L.reshape(BH, T)

    grid = (BH, T // BLOCK_Q)
    BLOCK_D = triton.next_power_of_2(d)
    _fwd_kernel[grid](
        Qf, Kf, Vf, Of, Lf,
        q_row_offsets, q_col_indices,
        sm_scale, T,
        Qf.stride(0), Qf.stride(1), Qf.stride(2), Lf.stride(0),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, D=d,
    )
    return O, L


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.

    Inputs:
      Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
      L              (B, H, T)            fp32   the forward residual
      k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
      k_col_indices  (nnz,)               int32  the query blocks i that attend it
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
      q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
      sm_scale       float
      BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
      dQ, dK, dV     (B, H, T, d)         fp16

    See ALGORITHMS.md sec 4 for the five gradient equations.

    TODO: implement.
    """
    B, H, T, d = Q.shape
    BH = B * H
    Qf = Q.reshape(BH, T, d)
    Kf = K.reshape(BH, T, d)
    Vf = V.reshape(BH, T, d)
    Of = O.reshape(BH, T, d)
    dOf = dO.reshape(BH, T, d)
    Lf = L.reshape(BH, T)
    # calculate D = sum_dO_O for each query position
    D = (dO.to(torch.float32) * O.to(torch.float32)).sum(dim=-1)
    Df = D.reshape(BH, T)

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    dQf = dQ.reshape(BH, T, d)
    dKf = dK.reshape(BH, T, d)
    dVf = dV.reshape(BH, T, d)

    BLOCK_D = triton.next_power_of_2(d)
    st_b, st_t, st_d = Qf.stride()
    st_lb = Lf.stride(0)

    _bwd_dq_kernel[(BH, T // BLOCK_Q)](
        Qf, Kf, Vf, dOf, Lf, Df, dQf,
        q_row_offsets, q_col_indices,
        sm_scale, T, st_b, st_t, st_d, st_lb,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, D=d,
    )
    _bwd_dkv_kernel[(BH, T // BLOCK_K)](
        Qf, Kf, Vf, dOf, Lf, Df, dKf, dVf,
        k_row_offsets, k_col_indices,
        sm_scale, T, st_b, st_t, st_d, st_lb,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, D=d,
    )
    return dQ, dK, dV
