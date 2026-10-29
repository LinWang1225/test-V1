# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN-MLA on the sparse-MLA (DSA) backend — e.g. GLM-5.2 / DeepSeek-V3.2.

The dense KVarN-MLA hook lives in TritonMLAImpl (latent int4 TILE cache + fp16
tail pool + Hadamard/Sinkhorn flush). The sparse backend (FlashMLASparseImpl)
selects the top-`index_topk` KV slots per query with a DSA "lightning indexer"
and attends them with `flash_mla_sparse_fwd`, which gathers the chosen slots
INSIDE a compiled bf16 kernel from the FULL paged cache. KVarN stores the latent
int4, so it can't hand that kernel a bf16 view — it must gather+dequant the
selected slots into a bf16 workspace first (the same shape as the existing fp8
prefill upconvert workspace), then call the unchanged bf16 kernel.

Design (v1, group == page == 64 so every dense tile kernel applies unchanged):
  store  : rotate latent c@H, scatter into the fp16 tail pool (graph-safe);
           full 64-token tiles flushed to int4 in the metadata builder.
  attend : per-request topk -> GLOBAL cache slots (triton_convert..., preserves
           -1 padding) -> gather+dequant selected slots into a bf16 workspace
           (ROTATED latent | rope) -> remap indices to workspace offsets ->
           flash_mla_sparse_fwd. The QUERY's nope part is rotated (@H) and the
           OUTPUT un-rotated (@H) — NOT the [T*topk,512] KV — because q.c =
           (q@H).(c@H) and rotating q+out is ~8x fewer FLOPs than un-rotating KV.

The gather kernel (the only new code) is validated standalone in
scripts_kvarn_mla/sparse_gather_validate.py (cos 0.999998, -1 -> zeros, dual
pool/int4 source). Everything else REUSES the dense kernels verbatim:
_kvarn_mla_scatter_store_kernel, _kvarn_batched_flush, kvarn_mla_tile_layout.
"""
import os

import torch

import triton.language as tl
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLAImpl,
    _kvarn_mla_scatter_store_kernel,
    kvarn_mla_tile_layout,
)

logger = init_logger(__name__)


def _kvarn_group_from_dtype(dtype_str: str) -> int:
    """Page/tile group encoded in the dtype suffix: '..._g64' -> 64 else 128."""
    return 64 if str(dtype_str).endswith("_g64") else 128


@triton.jit
def _kvarn_mla_sparse_gather_kernel(
    Cache, GlobalSlots, B2S, PoolLat, PoolRope, Dst,
    stride_dst, stride_pl_b, stride_pl_t, stride_pr_b, stride_pr_t,
    L: tl.constexpr, RPD: tl.constexpr, GRP: tl.constexpr, RECB: tl.constexpr,
    SC_OFF: tl.constexpr, ZP_OFF: tl.constexpr, SR_OFF: tl.constexpr,
    RP_OFF: tl.constexpr, NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """grid (T*topk,). One workspace row r = t*topk + j. GlobalSlots[r] is the
    global cache slot for that selection (-1 = pad/skip). phys = slot//GRP,
    intra = slot%GRP. Reads the ROTATED latent+rope from the fp16 tail pool
    (block holds a pool slot) or by dequanting the int4 TILE record (flushed
    block) — identical record-reading arithmetic to the dense gather, only the
    coordinate source differs (a global slot instead of a per-seq position).
    Writes [lat_rot(L) | rope(RPD)]; the query is rotated, so latent stays
    rotated (no un-rotation here)."""
    r = tl.program_id(0)
    slot = tl.load(GlobalSlots + r)
    offs_l = tl.arange(0, L)
    offs_r = tl.arange(0, RPD)
    dbase = Dst + r * stride_dst
    if slot < 0:
        tl.store(dbase + offs_l, tl.zeros([L], dtype=Dst.dtype.element_ty))
        tl.store(dbase + L + offs_r, tl.zeros([RPD], dtype=Dst.dtype.element_ty))
        return
    phys = slot // GRP
    intra = (slot % GRP).to(tl.int64)
    in_range = (phys >= 0) & (phys < NUM_BLOCKS_LOOKUP)
    pslot = tl.load(B2S + phys, mask=in_range, other=-1)
    if pslot >= 0:
        lat = tl.load(PoolLat + pslot.to(tl.int64) * stride_pl_b
                      + intra * stride_pl_t + offs_l).to(tl.float32)
        rope = tl.load(PoolRope + pslot.to(tl.int64) * stride_pr_b
                       + intra * stride_pr_t + offs_r).to(tl.float32)
    else:
        base = phys.to(tl.int64) * RECB
        pk = tl.load(Cache + base + intra * (L // 2)
                     + tl.arange(0, L // 2)).to(tl.uint32)
        sc = tl.load((Cache + base + SC_OFF).to(tl.pointer_type(tl.float16))
                     + offs_l).to(tl.float32)
        zp = tl.load((Cache + base + ZP_OFF).to(tl.pointer_type(tl.float16))
                     + offs_l).to(tl.float32)
        pt = tl.load((Cache + base + SR_OFF).to(tl.pointer_type(tl.float16))
                     + intra).to(tl.float32)
        lat = tl.interleave((pk & 0xF).to(tl.float32),
                            ((pk >> 4) & 0xF).to(tl.float32))
        lat = (lat * sc + zp) * pt
        rope = tl.load((Cache + base + RP_OFF).to(tl.pointer_type(tl.float16))
                       + intra * RPD + offs_r).to(tl.float32)
    tl.store(dbase + offs_l, lat.to(Dst.dtype.element_ty))
    tl.store(dbase + L + offs_r, rope.to(Dst.dtype.element_ty))


class FlashMLASparseKVarNImpl(FlashMLASparseImpl):
    """FlashMLASparseImpl + KVarN latent int4 quant. Transparent pass-through for
    non-kvarn (fp8/bf16) dtypes — get_impl_cls returns this class unconditionally
    and every KVarN path is gated on `self._is_kvarn`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_kvarn = str(self.kv_cache_dtype).startswith("kvarn_")
        if not self._is_kvarn:
            return
        self.qk_rope_head_dim = int(kwargs["qk_rope_head_dim"])
        self.vllm_config = get_current_vllm_config()
        self._kvarn_bits = 4
        self._kvarn_group = _kvarn_group_from_dtype(self.kv_cache_dtype)
        assert self._kvarn_group == 64, (
            "sparse-MLA KVarN v1 supports only group 64 (page==tile==64); "
            f"got {self._kvarn_group}. Use kv-cache-dtype=kvarn_k4v2_g64.")
        assert self.kv_lora_rank == 512 and self.qk_rope_head_dim == 64, (
            f"sparse KVarN expects 512+64 latent, got "
            f"{self.kv_lora_rank}+{self.qk_rope_head_dim}")
        # per-layer fp16 tail pool (in-progress blocks) + this layer's int4 cache
        self._kvarn_lat_pool = None    # [POOL, G, R] fp16 (rotated)
        self._kvarn_rope_pool = None   # [POOL, G, ROPE] fp16
        self._kv_cache_ref = None
        self._kvarn_H = None
        self._kvarn_Hb = None
        cls = type(self)
        if not hasattr(cls, "_kvarn_b2slot_dict"):
            cls._kvarn_b2slot_dict = {}     # block_id -> pool slot
            cls._kvarn_free_slots = None
            cls._kvarn_pool_size = 0
            cls._kvarn_b2slot_t = {}        # device -> int32[num_blocks]
            cls._kvarn_impls = []
            cls._kvarn_fill = {}
        cls._kvarn_impls.append(self)

    # ── shared helpers (mirror TritonMLAImpl) ───────────────────────────────
    def _kvarn_hadamard(self, n, device, dtype):
        if self._kvarn_H is None:
            H = torch.ones(1, 1, dtype=torch.float32)
            while H.shape[0] < n:
                H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
            self._kvarn_H = (H / n ** 0.5).to(device=device, dtype=torch.float32)
        return self._kvarn_H

    def _kvarn_hadamard_act(self, n, device, dtype):
        if self._kvarn_Hb is None or self._kvarn_Hb.dtype != dtype:
            self._kvarn_Hb = self._kvarn_hadamard(n, device, dtype).to(dtype)
        return self._kvarn_Hb

    def _ensure_kvarn_pool(self, device, num_blocks, pool_slots=None):
        """Allocate per-layer fp16 tail pool + class allocator + GPU block->slot
        lookup. b2s_t is sized ONCE to the full block count (its pointer is baked
        into the captured CUDA graph -> must never realloc)."""
        if torch.cuda.is_current_stream_capturing():
            return
        cls = type(self)
        R, ROPE, G = self.kv_lora_rank, self.qk_rope_head_dim, self._kvarn_group
        if pool_slots is None:
            pool_slots = int(os.environ.get("KVARN_MLA_POOL_SLOTS", "0")) or 512
        if self._kvarn_lat_pool is None:
            self._kvarn_lat_pool = torch.zeros(
                pool_slots, G, R, dtype=torch.float16, device=device)
            self._kvarn_rope_pool = torch.zeros(
                pool_slots, G, ROPE, dtype=torch.float16, device=device)
            if cls._kvarn_free_slots is None or cls._kvarn_pool_size != pool_slots:
                cls._kvarn_free_slots = list(range(pool_slots - 1, -1, -1))
                cls._kvarn_pool_size = pool_slots
                cls._kvarn_b2slot_dict.clear()
        try:
            total_blocks = self.vllm_config.cache_config.num_gpu_blocks or 0
        except Exception:
            total_blocks = 0
        num_blocks = max(num_blocks, 1024, total_blocks)
        existing = cls._kvarn_b2slot_t.get(device)
        if existing is None or existing.shape[0] < num_blocks:
            if existing is not None and torch.cuda.is_current_stream_capturing():
                return
            new_t = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
            if existing is not None:
                new_t[:existing.shape[0]] = existing
            for bid, slot in cls._kvarn_b2slot_dict.items():
                if bid < num_blocks:
                    new_t[bid] = slot
            cls._kvarn_b2slot_t[device] = new_t

    # ── store (graph-safe scatter into the fp16 tail pool) ──────────────────
    def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                           kv_cache_dtype, k_scale):
        if not self._is_kvarn:
            return super().do_kv_cache_update(
                kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)
        if kv_cache.numel() == 0:
            return
        self._kv_cache_ref = kv_cache
        Tn = kv_c_normed.shape[0]
        if Tn == 0:
            return
        if self._kvarn_lat_pool is None:
            self._ensure_kvarn_pool(kv_cache.device, kv_cache.shape[0])
        # Store the latent ALREADY ROTATED (c@H) so pool + int4 share one frame.
        Hb = self._kvarn_hadamard_act(self.kv_lora_rank, kv_c_normed.device,
                                      kv_c_normed.dtype)
        lat = (kv_c_normed @ Hb).to(torch.float16)
        rope = k_pe.reshape(Tn, -1).to(torch.float16)
        b2s = type(self)._kvarn_b2slot_t[kv_cache.device]
        _kvarn_mla_scatter_store_kernel[(Tn,)](
            lat, rope, slot_mapping.flatten().long(), b2s,
            self._kvarn_lat_pool, self._kvarn_rope_pool,
            lat.stride(0), rope.stride(0),
            self._kvarn_lat_pool.stride(0), self._kvarn_lat_pool.stride(1),
            self._kvarn_rope_pool.stride(0), self._kvarn_rope_pool.stride(1),
            R=self.kv_lora_rank, ROPE=self.qk_rope_head_dim,
            GROUP=self._kvarn_group, NUM_BLOCKS_LOOKUP=b2s.shape[0],
        )

    # ── attend (gather+dequant -> bf16 workspace -> sparse flash kernel) ─────
    def _kvarn_gather_attend(self, q, kv_cache, topk_indices, attn_metadata):
        global_slots = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token, attn_metadata.block_table,
            topk_indices, BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1])
        T, topk = global_slots.shape
        R, ROPE, G = self.kv_lora_rank, self.qk_rope_head_dim, self._kvarn_group
        NB, SC, ZP, SR, RP, REC, _ = kvarn_mla_tile_layout(R, ROPE, G,
                                                           self._kvarn_bits)
        b2s = type(self)._kvarn_b2slot_t[kv_cache.device]
        cache_flat = kv_cache.reshape(-1)
        pool_lat, pool_rope = self._kvarn_lat_pool, self._kvarn_rope_pool
        out = q.new_empty((T, self.num_heads, R))
        # Bound the [n*topk, 576] workspace (prefill chunks can be large): process
        # query tokens in row-blocks. Decode batch=256 -> 1 block ~600MB.
        cap_rows = int(os.environ.get("KVARN_SPARSE_WS_ROWS", str(256 * 1024)))
        blk = max(1, cap_rows // max(topk, 1))
        for s in range(0, T, blk):
            e = min(s + blk, T)
            n = e - s
            gs = global_slots[s:e].reshape(-1).to(torch.int64).contiguous()
            ws = q.new_empty((n * topk, R + ROPE))
            _kvarn_mla_sparse_gather_kernel[(n * topk,)](
                cache_flat, gs, b2s, pool_lat, pool_rope, ws,
                ws.stride(0), pool_lat.stride(0), pool_lat.stride(1),
                pool_rope.stride(0), pool_rope.stride(1),
                L=R, RPD=ROPE, GRP=G, RECB=REC,
                SC_OFF=SC, ZP_OFF=ZP, SR_OFF=SR, RP_OFF=RP,
                NUM_BLOCKS_LOOKUP=b2s.shape[0],
            )
            ar = torch.arange(n * topk, device=gs.device,
                              dtype=torch.int32).view(n, topk)
            remapped = torch.where(gs.view(n, topk) >= 0, ar,
                                   torch.full_like(ar, -1))
            out[s:e] = self._bf16_flash_mla_kernel(q[s:e], ws, remapped)
        return out

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if not self._is_kvarn:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        # Rotate the absorbed query's nope part into the KVarN frame, concat.
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            Hb = self._kvarn_hadamard_act(self.kv_lora_rank, ql_nope.device,
                                          ql_nope.dtype)
            ql_nope = (ql_nope @ Hb).to(ql_nope.dtype)
            qc = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, qc)
            q = qc
        else:
            Hb = self._kvarn_hadamard_act(self.kv_lora_rank, q.device, q.dtype)
            ql = (q[..., : self.kv_lora_rank] @ Hb).to(q.dtype)
            q = torch.cat([ql, q[..., self.kv_lora_rank:]], dim=-1)
        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        attn_out = self._kvarn_gather_attend(
            q, kv_c_and_k_pe_cache, topk_indices, attn_metadata)
        # Output comes back ROTATED (value frame == latent frame) -> un-rotate.
        Hb = self._kvarn_hadamard_act(self.kv_lora_rank, attn_out.device,
                                      attn_out.dtype)
        attn_out = (attn_out @ Hb).to(attn_out.dtype)
        return attn_out, None


class FlashMLASparseKVarNMetadataBuilder(FlashMLASparseMetadataBuilder):
    """FlashMLASparseMetadataBuilder + the KVarN tile-flush / pool slot lifecycle
    (mirrors TritonMLAMetadataBuilder.build). Runs OUTSIDE any captured region
    (between graph replays). Transparent for non-kvarn dtypes."""

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        cls = FlashMLASparseKVarNImpl
        impls = getattr(cls, "_kvarn_impls", [])
        if not impls or not getattr(impls[0], "_is_kvarn", False):
            return md
        G = impls[0]._kvarn_group

        slc = common_attn_metadata.seq_lens_cpu_upper_bound
        if slc is None:
            slc = common_attn_metadata.seq_lens.cpu()
        seq_lens_cpu = slc.tolist()
        B = len(seq_lens_cpu)
        n_tok = common_attn_metadata.num_actual_tokens

        # FAST-SKIP steady-state decode (no block boundary crossed this step).
        prev = cls.__dict__.get("_kvarn_prev_seqlens")
        if (not os.environ.get("KVARN_MLA_NOSKIP")
                and n_tok == B and prev is not None and len(prev) == B
                and all(seq_lens_cpu[i] == prev[i] + 1 and seq_lens_cpu[i] % G > 1
                        for i in range(B))):
            cls._kvarn_prev_seqlens = seq_lens_cpu
            return md
        cls._kvarn_prev_seqlens = seq_lens_cpu

        bt = common_attn_metadata.block_table_tensor
        device = bt.device
        block_table_cpu = bt.tolist()

        sched = self.vllm_config.scheduler_config
        pool_slots = int(os.environ.get("KVARN_MLA_POOL_SLOTS", "0")) or max(
            2 * sched.max_num_seqs
            + (sched.max_num_batched_tokens + G - 1) // G, 64)
        nb_hint = (getattr(self.vllm_config.cache_config, "num_gpu_blocks", None)
                   or 0)
        if nb_hint < 1024:
            nb_hint = 1024
        for row in block_table_cpu:
            for b in row:
                if b >= 0 and b + 1 > nb_hint:
                    nb_hint = b + 1
        for impl in impls:
            impl._ensure_kvarn_pool(device, nb_hint, pool_slots)
        b2s_t = cls._kvarn_b2slot_t[device]
        b2s_dict = cls._kvarn_b2slot_dict
        free = cls._kvarn_free_slots
        fill = cls._kvarn_fill

        qsl = getattr(common_attn_metadata, "query_start_loc_cpu", None)
        if qsl is not None:
            qsl = qsl.tolist()
            query_lens = [qsl[i + 1] - qsl[i] for i in range(B)]
        else:
            query_lens = [(n_tok // B if B else 1)] * B

        blocks_needed: set[int] = set()
        flush_seen: set[int] = set()
        flush_q: list[int] = []
        for b in range(B):
            row = block_table_cpu[b]
            sl = seq_lens_cpu[b]
            if not row or sl <= 0:
                continue
            committed = max(sl - query_lens[b], 0)
            for k in range(committed // G, min((sl - 1) // G, len(row) - 1) + 1):
                bid = row[k]
                if bid >= 0:
                    blocks_needed.add(bid)
                    fill[bid] = min(sl, (k + 1) * G) - k * G
            k = committed // G - 1
            while 0 <= k < len(row):
                bid = row[k]
                if bid < 0 or bid in flush_seen or bid not in b2s_dict:
                    break
                flush_seen.add(bid)
                flush_q.append(bid)
                k -= 1

        # RECLAIM finished/descheduled requests' slot-holding blocks.
        discard_ids: list[int] = []
        for bid in list(b2s_dict):
            if bid in blocks_needed or bid in flush_seen:
                continue
            if fill.get(bid, 0) >= G:
                flush_seen.add(bid)
                flush_q.append(bid)
            else:
                discard_ids.append(bid)

        if flush_q:
            iters = int(os.environ.get("KVARN_SINKHORN_ITERS", "16"))
            flush_list = []
            for impl in impls:
                if impl._kv_cache_ref is None:
                    continue
                for bid in flush_q:
                    slot = b2s_dict.get(bid)
                    if slot is not None:
                        flush_list.append((impl, bid, slot))
            # Reuse the dense batched flush verbatim (touches only impl.* attrs).
            TritonMLAImpl._kvarn_batched_flush(flush_list, iters)
        for bid in flush_q:
            slot = b2s_dict.pop(bid, None)
            fill.pop(bid, None)
            if slot is not None:
                free.append(slot)
                if bid < b2s_t.shape[0]:
                    b2s_t[bid] = -1
        for bid in discard_ids:
            slot = b2s_dict.pop(bid)
            fill.pop(bid, None)
            free.append(slot)
            if bid < b2s_t.shape[0]:
                b2s_t[bid] = -1

        # ALLOCATE slots for needed blocks lacking one.
        for bid in blocks_needed:
            if bid in b2s_dict:
                continue
            if not free:
                raise RuntimeError(
                    f"KVarN-MLA sparse pool exhausted (size={pool_slots}); "
                    f"raise KVARN_MLA_POOL_SLOTS")
            slot = free.pop()
            b2s_dict[bid] = slot
            if bid < b2s_t.shape[0]:
                b2s_t[bid] = slot
            if not os.environ.get("KVARN_MLA_NOZERO"):
                for impl in impls:
                    if impl._kvarn_lat_pool is not None:
                        impl._kvarn_lat_pool[slot].zero_()
                        impl._kvarn_rope_pool[slot].zero_()
        return md
