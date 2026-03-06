import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Simplified LSHAttention Module (from previous implementation)
class LSHAttention(nn.Module):
    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=8, dropout=0., return_attn=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.return_attn = return_attn

        self.dropout = nn.Dropout(dropout)
        self.to_qk = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def hash_vectors(self, n_buckets, vecs):
        batch_size, seqlen, dim = vecs.shape
        device = vecs.device

        assert n_buckets % 2 == 0
        rot_size = n_buckets

        rotations_shape = (1, dim, self.n_hashes, rot_size // 2)
        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        rotated_vecs = torch.einsum('btf,bfhi->bhti', vecs, random_rotations)
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        buckets = torch.argmax(rotated_vecs, dim=-1)

        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        return buckets

    def forward(self, x):
      batch_size, seqlen, dim = x.shape
      device = x.device

      # Calculate padding if necessary
      target_seqlen = (seqlen + self.bucket_size * 2 - 1) // (self.bucket_size * 2) * (self.bucket_size * 2)
      pad_len = target_seqlen - seqlen

      if pad_len > 0:
          # Pad the sequence length dimension with zeros
          x = F.pad(x, (0, 0, 0, pad_len), mode='constant', value=0)

      qk = self.to_qk(x)
      v = self.to_v(x)

      n_buckets = target_seqlen // self.bucket_size
      buckets = self.hash_vectors(n_buckets, qk)

      total_hashes = self.n_hashes
      ticker = torch.arange(total_hashes * target_seqlen, device=device).unsqueeze(0).expand_as(buckets)
      buckets_and_t = target_seqlen * buckets + (ticker % target_seqlen)
      buckets_and_t = buckets_and_t.detach()

      sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
      _, undo_sort = sticker.sort(dim=-1)

      st = (sticker % target_seqlen)
      sqk = batched_index_select(qk, st)
      sv = batched_index_select(v, st)

      chunk_size = total_hashes * n_buckets
      bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
      bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
      bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

      bq = bqk
      bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

      dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
      dots = self.dropout(dots)

      bo = torch.einsum('buij,buje->buie', dots, bv)
      so = torch.reshape(bo, (batch_size, -1, dim))

      o = batched_index_select(so, undo_sort)
      o = torch.reshape(o, (batch_size, total_hashes, target_seqlen, dim))
      out = torch.sum(o, dim=1)

      # Remove padding to match the original sequence length
      out = out[:, :seqlen, :]

      out = self.to_out(out)

      if self.return_attn:
          attn = torch.sum(dots, dim=1)
          return out, attn
      return out


# Helper Functions
def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(t, inds):
    batch_size, seqlen, dim = t.shape
    device = t.device
    inds = inds.reshape(batch_size, -1)
    inds = inds.unsqueeze(-1).expand(-1, -1, dim)
    return t.gather(1, inds)

# FeedForward Module
class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., activation=None, mult=4, glu=False):
        super().__init__()
        inner_dim = int(dim * mult)
        activation = nn.GELU() if activation is None else activation

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ViR(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=68,
        in_channels=3,
        dim=768,
        depth=12,
        heads=8,
        pool = 'cls',
        bucket_size=64,
        n_hashes=8,
        ff_mult=4,
        lsh_dropout=0.1,
        ff_dropout=0.1,
        emb_dropout=0.1,
        use_rezero=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Patch embedding
        patch_height = patch_width = patch_size
        patch_dim = in_channels * patch_height * patch_width
        num_patches = (img_size // patch_size) ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional embeddings and cls token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Reformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LSHAttention(dim, heads, bucket_size, n_hashes, dropout=lsh_dropout)
            ff = FeedForward(dim, dropout=ff_dropout, mult=ff_mult)

            if use_rezero:
                attn = ReZero(attn)
                ff = ReZero(ff)
            else:
                attn = PreNorm(dim, attn)
                ff = PreNorm(dim, ff)

            self.layers.append(nn.ModuleList([attn, ff]))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.to_patch_embedding(x)  # (batch_size, num_patches, dim)

        # Add cls token
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, dim)

        # Add positional embeddings
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Apply Reformer layers
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    def extract_features(self, x):
        # Patch embedding
        x = self.to_patch_embedding(x)  # (batch_size, num_patches, dim)

        # Add cls token
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, dim)

        # Add positional embeddings
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Apply Reformer layers
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x

# PreNorm and ReZero Wrappers
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, x, **kwargs):
        return x * self.alpha + self.fn(x, **kwargs)