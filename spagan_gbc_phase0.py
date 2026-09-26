"""
spagan_gbc_phase0.py — Phase 0 Training Script for GEN-GBC.

Trains a shortest-path attention surrogate for exact group betweenness
centrality (GBC) on one already loaded, undirected, unweighted graph.

Runs the full Phase 0 pipeline:
  1. Sample 1,000 different groups of k=10 vertices on the full graph.
  2. Label every group using the supplied exact_gbc.cpp evaluator.
  3. Split groups into 800 training and 200 validation examples.
  4. Train a five-layer shortest-path attention encoder and group decoder.
  5. Save the best checkpoint and validation report.

Architecture/training follow Hephaestus's paper where its supplemental code
has different settings. Path sampling and the set decoder adapt the method to
GBC. Labels use the exact C++ program's ordered-pair, internal-node objective.
The path encoder uses two-level attention by hop length; it does not perform
the original SPAGAN paper's iterative regeneration of learned-weight paths.
This module does not load a dataset: the final experiment runner supplies
GraphData(edge_index=[2, E], num_nodes=N, edge_weight=None).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from core.types import GraphData
except ImportError:
    try:
        from src.core.types import GraphData
    except ImportError:
        GraphData = Any  # Standalone smoke test without GEN-CIM package.


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SPAGANTrainConfig:
    """Model and optimizer settings for the fixed-k surrogate.

    Five attention layers, 512 hidden units, eight heads, Adam at 5e-4,
    batch size 256, 3,000 epochs, and Huber loss follow the Hephaestus paper.
    Early stopping can be enabled explicitly for smaller experiments.
    """

    k: int = 10
    max_epochs: int = 3000
    patience: int | None = None
    batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    dropout: float = 0.2
    max_hops: int = 3
    paths_per_hop: int = 4
    seed: int = 42
    hidden_dim: int = 512
    n_layers: int = 5
    n_heads: int = 8


@dataclass(frozen=True)
class Phase0Config:
    """Sampling, exact-labeling, and 80/20 split for the first pilot."""

    k: int = 10
    num_groups: int = 1000
    exact_batch_size: int = 50
    exact_threads: int = 1
    seed: int = 42
    train_fraction: float = 0.8
    training: SPAGANTrainConfig = SPAGANTrainConfig()

# ═══════════════════════════════════════════════════════════════════════════════
#  Exact GBC labels and reproducible train/validation split
# ═══════════════════════════════════════════════════════════════════════════════


def _edges(graph: GraphData) -> tuple[list[tuple[int, int]], int]:
    """Canonical undirected, unweighted edge list using GraphData node IDs."""
    n = int(graph.num_nodes)
    if n < 1:
        raise ValueError("graph must have vertices")
    if getattr(graph, "edge_weight", None) is not None:
        raise ValueError("Phase 0 requires an unweighted graph (edge_weight=None)")
    edge_index = graph.edge_index.detach().cpu()
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("graph.edge_index must have shape [2, E]")
    pairs = edge_index.t().tolist()
    edges: set[tuple[int, int]] = set()
    for u, v in pairs:
        u, v = int(u), int(v)
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge_index contains a vertex outside [0, num_nodes)")
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return sorted(edges), n


def random_groups(num_nodes: int, k: int = 10, count: int = 1000,
                  seed: int = 42) -> list[tuple[int, ...]]:
    """Draw distinct uniform k-subsets; no vertex repeats within a set.

    Each draw uses ``random.sample`` without replacement. Duplicate groups
    are discarded, then groups are sorted to make dataset IDs reproducible.
    """
    if k < 1 or count < 1 or num_nodes < k:
        raise ValueError("require 1 <= k <= num_nodes and count >= 1")
    if math.comb(num_nodes, k) < count:
        raise ValueError("not enough distinct k-subsets for requested count")
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    while len(seen) < count:
        seen.add(tuple(sorted(rng.sample(range(num_nodes), k))))
    # Sort to make ID assignment deterministic independently of set iteration.
    return sorted(seen)


def _write_graph(path: Path, edges: Sequence[tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")


def graph_fingerprint(graph: GraphData) -> str:
    """Hash graph topology together with internal node count and ID ordering."""
    edges, n = _edges(graph)
    digest = hashlib.sha256()
    digest.update(f"n={n}\n".encode())
    for u, v in edges:
        digest.update(f"{u},{v}\n".encode())
    return digest.hexdigest()


def compile_exact_gbc(source_path: str | Path, binary_path: str | Path) -> Path:
    """Build the user's exact_gbc.cpp with C++17 and OpenMP if needed."""
    source, binary = Path(source_path).resolve(), Path(binary_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    binary.parent.mkdir(parents=True, exist_ok=True)
    if binary.exists() and binary.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return binary
    cmd = ["g++", "-O2", "-std=c++17", "-fopenmp", str(source), "-o", str(binary)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return binary


def create_exact_dataset(
    graph: GraphData, exact_binary: str | Path, output_dir: str | Path, *,
    k: int = 10, count: int = 1000, seed: int = 42,
    exact_batch_size: int = 50, threads: int = 1,
) -> Path:
    """Evaluate complete-graph exact GBC for sampled groups; resume batches.

    Parameters
    ----------
    graph : GraphData
        Undirected unweighted graph with contiguous node IDs.
    exact_binary : str | Path
        Compiled version of the user's standalone exact_gbc.cpp.
    output_dir : str | Path
        Directory for exported graph, labels, and objective manifest.
    k, count, seed : int
        Group size, number of unique groups, and sampling seed.
    exact_batch_size, threads : int
        Number of groups per C++ invocation and OpenMP thread count.

    Returns
    -------
    Path
        JSONL file containing group IDs, member nodes, and raw exact GBC.

    Each C++ invocation traverses the whole graph. Larger batches reduce
    repeated traversals, while completed batches remain safe to resume.
    """
    if exact_batch_size < 1 or threads < 1:
        raise ValueError("exact_batch_size and threads must be positive")
    edges, n = _edges(graph)
    groups = random_groups(n, k, count, seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "graph_internal_ids.txt"
    rows_path = output_dir / f"spagan_gbc_k{k}_labels.jsonl"
    meta_path = output_dir / f"spagan_gbc_k{k}_manifest.json"
    exact_binary = Path(exact_binary).resolve()
    if not exact_binary.is_file():
        raise FileNotFoundError(exact_binary)
    binary_sha256 = hashlib.sha256(exact_binary.read_bytes()).hexdigest()
    manifest = {"num_nodes": n, "graph_sha256": graph_fingerprint(graph), "directed": False,
                "weighted": False, "k": k, "count": count, "seed": seed,
                "exact_binary_sha256": binary_sha256,
                "normalized": False, "endpoints_counted": False,
                "pairs_with_group_endpoints_included": True,
                "pair_domain": "all_distinct_ordered_pairs",
                "coverage": "at_least_one_internal_group_node"}
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("cached labels belong to another graph/config; use a new output_dir")
    elif rows_path.exists():
        raise ValueError("labels exist but manifest is missing; use a new output_dir")
    else:
        meta_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_graph(graph_path, edges)
    existing: dict[int, dict[str, Any]] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            idx = int(row["id"])
            if idx in existing or idx >= count or tuple(row["nodes"]) != groups[idx]:
                raise ValueError("invalid or duplicate cached label")
            existing[idx] = row
    label_start = time.perf_counter()
    for start in range(0, count, exact_batch_size):
        indices = [i for i in range(start, min(start + exact_batch_size, count))
                   if i not in existing]
        if not indices:
            continue
        groups_path = output_dir / "exact_batch_groups.txt"
        with groups_path.open("w", encoding="utf-8") as f:
            for i in indices:
                f.write(f"g{i:06d}: {' '.join(map(str, groups[i]))}\n")
        cmd = [str(exact_binary), "--graph", str(graph_path), "--num-nodes", str(n),
               "--groups-file", str(groups_path), "--group-ids", "external",
               "--threads", str(threads)]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"exact_gbc failed on batch {start}: {result.stderr[-3000:]}")
        output = json.loads(result.stdout)
        for key in ("directed", "weighted", "normalized", "endpoints_counted",
                    "pairs_with_group_endpoints_included", "pair_domain", "coverage"):
            if output.get(key) != manifest[key]:
                raise RuntimeError(f"exact_gbc objective mismatch: {key}={output.get(key)!r}")
        if int(output.get("num_nodes", -1)) != n:
            raise RuntimeError("exact_gbc read a different graph size")
        returned = {row["method"]: row for row in output["results"]}
        if len(returned) != len(indices):
            raise RuntimeError("exact_gbc returned the wrong number of groups")
        with rows_path.open("a", encoding="utf-8") as f:
            for i in indices:
                row = returned.get(f"g{i:06d}")
                if row is None:
                    raise RuntimeError("exact_gbc result is missing a group")
                if tuple(sorted(map(int, row["nodes"]))) != groups[i]:
                    raise RuntimeError("exact_gbc node IDs do not match GraphData IDs")
                score = float(row["raw_gbc"])
                if not math.isfinite(score) or score < -1e-7:
                    raise RuntimeError("exact_gbc returned an invalid raw score")
                record = {"id": i, "nodes": list(groups[i]), "raw_gbc": max(0., score)}
                f.write(json.dumps(record) + "\n")
                existing[i] = record
        elapsed = time.perf_counter() - label_start
        print(f"[Phase0] Exact GBC: {len(existing)}/{count} sets, "
              f"elapsed={elapsed:.1f}s")
    return rows_path


def train_validation_split(rows_path: str | Path, *, seed: int = 42,
                           train_fraction: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Split exact-labeled groups by ID, with no overlap between partitions."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    records = [json.loads(s) for s in Path(rows_path).read_text(encoding="utf-8").splitlines()]
    if len(records) < 2:
        raise ValueError("need at least two labelled groups")
    records.sort(key=lambda r: int(r["id"]))
    rng = random.Random(seed)
    rng.shuffle(records)
    split = max(1, min(len(records) - 1, int(len(records) * train_fraction)))
    return records[:split], records[split:]

# ═══════════════════════════════════════════════════════════════════════════════
#  Node features and shortest-path attention model
# ═══════════════════════════════════════════════════════════════════════════════


def _adjacency(edge_index: Tensor, num_nodes: int) -> list[list[int]]:
    """Build undirected neighbors from GEN-CIM's COO edge representation."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be [2, E]")
    adj = [set() for _ in range(num_nodes)]
    for u, v in edge_index.detach().cpu().t().tolist():
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("edge_index contains an invalid vertex")
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return [sorted(x) for x in adj]


def degree_features(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Compute three GEN-CIM-style structural features for each vertex.

    Returns
    -------
    Tensor [N, 3]
        ``[log1p(degree), degree/dmax, 1/sqrt(degree+1)]``.
    """
    adj = _adjacency(edge_index, num_nodes)
    deg = torch.tensor([len(neighbors) for neighbors in adj], dtype=torch.float32,
                       device=edge_index.device)
    dmax = deg.max().clamp_min(1)
    return torch.stack((torch.log1p(deg), deg / dmax,
                        torch.rsqrt(deg + 1)), dim=-1)


@dataclass
class ShortestPathIndex:
    """Shortest-path attention neighborhoods built once for a fixed graph."""

    source: Tensor                  # [P]
    nodes: Tensor                   # [P, max_hops], excludes source; padded with 0
    mask: Tensor                    # [P, max_hops], true iff node is on path
    num_nodes: int

    def to(self, device: torch.device | str) -> "ShortestPathIndex":
        return ShortestPathIndex(self.source.to(device), self.nodes.to(device),
                                 self.mask.to(device), self.num_nodes)


def sample_shortest_paths(
    edge_index: Tensor, num_nodes: int, *, max_hops: int = 3,
    paths_per_hop: int = 4, seed: int = 42,
) -> ShortestPathIndex:
    """One BFS shortest path per selected target, capped per hop and source.

    This is topology-only preprocessing.  It does not use GBC labels or select
    one path as the ground truth for GBC.  Multiple shortest paths may exist;
    a reproducible BFS path is an *attention neighborhood*, not a GBC label.
    """
    if max_hops < 1 or paths_per_hop < 1:
        raise ValueError("max_hops and paths_per_hop must be positive")
    adj = _adjacency(edge_index, num_nodes)
    rng = random.Random(seed)
    src_list: list[int] = []
    node_list: list[list[int]] = []
    mask_list: list[list[bool]] = []
    for source in range(num_nodes):
        parent = {source: -1}
        depth = {source: 0}
        targets: list[list[int]] = [[] for _ in range(max_hops + 1)]
        queue = deque([source])
        while queue:
            u = queue.popleft()
            if depth[u] == max_hops:
                continue
            for v in adj[u]:
                if v in parent:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                targets[depth[v]].append(v)
                queue.append(v)
        for hops in range(1, max_hops + 1):
            candidates = targets[hops]
            selected = (rng.sample(candidates, paths_per_hop)
                        if len(candidates) > paths_per_hop else candidates)
            for target in selected:
                path: list[int] = []
                at = target
                while at != source:
                    path.append(at)
                    at = parent[at]
                path.reverse()
                src_list.append(source)
                node_list.append(path + [0] * (max_hops - hops))
                mask_list.append([True] * hops + [False] * (max_hops - hops))
    if not src_list:
        return ShortestPathIndex(torch.empty(0, dtype=torch.long),
                                 torch.empty((0, max_hops), dtype=torch.long),
                                 torch.empty((0, max_hops), dtype=torch.bool), num_nodes)
    return ShortestPathIndex(torch.tensor(src_list, dtype=torch.long),
                             torch.tensor(node_list, dtype=torch.long),
                             torch.tensor(mask_list, dtype=torch.bool), num_nodes)


class ShortestPathAttentionLayer(nn.Module):
    """Hierarchical shortest-path attention for one graph encoder layer.

    First aggregate paths of equal length for each source node, then attend
    over the resulting length-specific vectors and the source itself. This
    follows the two-level path aggregation described by SPAGAN, using mean
    node pooling for each path. The path index is a fixed topology input.
    """

    def __init__(self, width: int = 512, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.head_dim = heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.length_query = nn.Linear(width, width, bias=False)
        self.length_key = nn.Linear(width, width, bias=False)
        self.self_value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, paths: ShortestPathIndex) -> Tensor:
        if paths.source.numel() == 0:
            return h
        n = paths.num_nodes
        max_hops = paths.nodes.size(1)

        # --- Pool node features along each shortest path ---
        mask = paths.mask.unsqueeze(-1)
        path_h = (h[paths.nodes] * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        # --- Level 1: attention among paths of the same hop length ---
        q = self.query(h[paths.source]).view(-1, self.heads, self.head_dim)
        key = self.key(path_h).view(-1, self.heads, self.head_dim)
        value = self.value(path_h).view(-1, self.heads, self.head_dim)
        logits = (q * key).sum(-1) * (self.head_dim ** -0.5)
        hop = paths.mask.sum(dim=1).long() - 1
        length_id = paths.source * max_hops + hop
        index = length_id[:, None].expand(-1, self.heads)
        bins = n * max_hops
        max_per_length = h.new_full((bins, self.heads), -torch.inf)
        max_per_length.scatter_reduce_(0, index, logits, reduce="amax",
                                       include_self=True)
        weights = torch.exp(logits - max_per_length[length_id])
        denom = h.new_zeros((bins, self.heads))
        denom.scatter_add_(0, index, weights)
        alpha = weights / denom[length_id].clamp_min(1e-12)
        pooled = h.new_zeros((bins, self.heads, self.head_dim))
        pooled.index_add_(0, length_id,
                          value * self.dropout(alpha).unsqueeze(-1))
        pooled = pooled.view(n, max_hops, self.heads, self.head_dim)
        valid = (denom.sum(dim=1) > 0).view(n, max_hops)

        # --- Level 2: attention across hop lengths, including self ---
        self_h = self.self_value(h).view(n, 1, self.heads, self.head_dim)
        candidates = torch.cat((self_h, pooled), dim=1)
        valid = torch.cat((torch.ones((n, 1), dtype=torch.bool, device=h.device),
                           valid), dim=1)
        q_length = self.length_query(h).view(n, 1, self.heads, self.head_dim)
        k_length = self.length_key(candidates.reshape(n, max_hops + 1, -1))
        k_length = k_length.view(n, max_hops + 1, self.heads, self.head_dim)
        length_logits = (q_length * k_length).sum(-1) * (self.head_dim ** -0.5)
        length_logits = length_logits.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        beta = torch.softmax(length_logits, dim=1)
        aggregated = (self.dropout(beta).unsqueeze(-1) * candidates).sum(dim=1)
        out = self.output(aggregated.reshape(n, -1))
        return self.norm(h + self.dropout(F.elu(out)))


class SPAGANGBC(nn.Module):
    """Graph and unordered k-set -> nonnegative predicted raw exact GBC.

    The paper reports five shortest-path GAT layers, width 512, eight heads;
    its supplemental ZIP instead uses ordinary GATConv. Here a path-attention
    encoder follows the paper's stated dimensions and hierarchical path
    aggregation, with a permutation-invariant set decoder needed for GBC.
    BFS path sampling is bounded; exact labels include all shortest paths.
    Learned-weight iterative path regeneration is not specified in the
    Hephaestus training pseudocode and is not performed here.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 512,
                 n_layers: int = 5, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.config = dict(input_dim=input_dim, hidden_dim=hidden_dim,
                           n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.path_layers = nn.ModuleList(
            ShortestPathAttentionLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        )
        # Self-attention across the k selected vertices models overlap between
        # members.  Mean and max pooling make the output order-invariant.
        self.group_attention = nn.MultiheadAttention(hidden_dim, n_heads,
                                                      dropout=dropout, batch_first=True)
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Softplus(),
        )

    def encode(self, node_features: Tensor, paths: ShortestPathIndex) -> Tensor:
        if node_features.ndim != 2 or node_features.size(1) != self.config["input_dim"]:
            raise ValueError("node_features must be [N, input_dim]")
        if node_features.size(0) != paths.num_nodes:
            raise ValueError("path index and node_features refer to different graphs")
        h = F.elu(self.input_projection(node_features))
        for layer in self.path_layers:
            h = layer(h, paths)
        return h

    def score_encoded(self, embeddings: Tensor, groups: Tensor) -> Tensor:
        if groups.ndim != 2 or groups.size(1) < 1:
            raise ValueError("groups must be [B, k] with k >= 1")
        if bool(((groups < 0) | (groups >= embeddings.size(0))).any()):
            raise ValueError("group contains an invalid vertex")
        if any(len(set(row)) != groups.size(1) for row in groups.tolist()):
            raise ValueError("each group must contain k distinct vertices")
        members = embeddings[groups]
        attn, _ = self.group_attention(members, members, members, need_weights=False)
        members = self.group_norm(members + attn)
        pooled = torch.cat((members.mean(1), members.max(1).values,
                            embeddings.mean(0).expand(groups.size(0), -1)), dim=-1)
        return self.regressor(pooled).squeeze(-1)

    def forward(self, node_features: Tensor, paths: ShortestPathIndex,
                groups: Tensor) -> Tensor:
        return self.score_encoded(self.encode(node_features, paths), groups)

    @torch.no_grad()
    def predict_raw(self, node_features: Tensor, paths: ShortestPathIndex,
                    groups: Tensor) -> Tensor:
        """One graph encoding, batched groups; returns approximate *raw* GBC."""
        self.eval()
        return self.forward(node_features, paths, groups)

# ═══════════════════════════════════════════════════════════════════════════════
#  SPAGAN-GBC training and inference
# ═══════════════════════════════════════════════════════════════════════════════


def _records_to_tensors(records: Sequence[dict], k: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return member IDs and *raw* exact-GBC targets without transforms."""
    if any(len(r["nodes"]) != k for r in records):
        raise ValueError(f"every label must be for exactly k={k} vertices")
    groups = torch.tensor([r["nodes"] for r in records], dtype=torch.long, device=device)
    raw = torch.tensor([r["raw_gbc"] for r in records], dtype=torch.float32, device=device)
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise ValueError("GBC labels must be finite and nonnegative")
    return groups, raw


def train_spagan_gbc(graph: GraphData, labels_path: str | Path,
                     checkpoint_path: str | Path, *,
                     config: SPAGANTrainConfig = SPAGANTrainConfig(),
                     device: str | torch.device = "cpu",
                     train_fraction: float = 0.8) -> dict:
    """Train SPAGAN-GBC and save the best validation checkpoint.

    Parameters
    ----------
    graph : GraphData
        Same graph and node mapping used to generate the exact labels.
    labels_path : str | Path
        JSONL file produced by ``create_exact_dataset``.
    checkpoint_path : str | Path
        Destination for the best model state and its graph fingerprint.
    config : SPAGANTrainConfig
        Encoder and optimizer settings.
    device : str | torch.device
        Training device, typically ``cuda`` or ``cpu``.
    train_fraction : float
        Fraction of labeled groups used for training (default 0.8).

    Returns
    -------
    dict
        Checkpoint path, epoch, train/validation counts and hold-out metrics.

    Huber loss is computed directly between predicted and exact raw-GBC
    scores, following Hephaestus's loss choice on its original target scale.
    """
    device = torch.device(device)
    torch.manual_seed(config.seed)
    manifest_path = Path(labels_path).with_name(
        f"spagan_gbc_k{config.k}_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = graph_fingerprint(graph)
    if manifest["graph_sha256"] != fingerprint or manifest["k"] != config.k:
        raise ValueError("training labels do not match the current graph and k")
    if (manifest.get("pair_domain") != "all_distinct_ordered_pairs"
            or manifest.get("coverage") != "at_least_one_internal_group_node"
            or manifest.get("normalized") is not False):
        raise ValueError("labels use a different exact-GBC objective")
    train_rows, val_rows = train_validation_split(labels_path, seed=config.seed,
                                                  train_fraction=train_fraction)
    if len(train_rows) < 2 or len(val_rows) < 1:
        raise ValueError("train/validation split is too small")
    train_groups, train_y = _records_to_tensors(train_rows, config.k, device)
    val_groups, val_y = _records_to_tensors(val_rows, config.k, device)
    edge_index = graph.edge_index
    x = degree_features(edge_index, graph.num_nodes).to(device)
    paths = sample_shortest_paths(edge_index, graph.num_nodes,
                                  max_hops=config.max_hops,
                                  paths_per_hop=config.paths_per_hop,
                                  seed=config.seed).to(device)
    model = SPAGANGBC(hidden_dim=config.hidden_dim, n_layers=config.n_layers,
                      n_heads=config.n_heads, dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best_loss, best_epoch, stale = math.inf, -1, 0
    best_state: dict | None = None
    train_start = time.perf_counter()
    for epoch in range(config.max_epochs):
        model.train()
        ordering = torch.randperm(len(train_rows), device=device)
        for idx in ordering.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, paths, train_groups[idx])
            loss = F.huber_loss(pred, train_y[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            h = model.encode(x, paths)
            preds = torch.cat([model.score_encoded(h, g)
                               for g in val_groups.split(config.batch_size)])
            val_loss = float(F.huber_loss(preds, val_y).item())
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = copy.deepcopy({key: val.detach().cpu()
                                        for key, val in model.state_dict().items()})
        else:
            stale += 1
            if config.patience is not None and stale >= config.patience:
                break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Phase0] epoch={epoch + 1} val_huber_raw={val_loss:.6f} "
                  f"best={best_loss:.6f}")
    assert best_state is not None
    # The selected checkpoint is determined solely by validation Huber loss.
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "model_config": model.config,
                "k": config.k, "num_nodes": int(graph.num_nodes),
                "graph_sha256": fingerprint,
                "path_config": {"max_hops": config.max_hops,
                                "paths_per_hop": config.paths_per_hop,
                                "seed": config.seed},
                "best_epoch": best_epoch, "validation_huber_raw": best_loss}, checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "best_epoch": best_epoch,
            "validation_huber_raw": best_loss, "train_count": len(train_rows),
            "validation_count": len(val_rows),
            "training_seconds": round(time.perf_counter() - train_start, 2)}


class SPAGANScorer:
    """Load a trained checkpoint and predict raw GBC for complete k-sets.

    Graph embeddings are cached once. A checkpoint is tied to the exact graph
    topology and internal node IDs used during Phase 0 training.
    """

    def __init__(self, graph: GraphData, checkpoint_path: str | Path,
                 device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.k = int(state["k"])
        if graph.num_nodes != int(state["num_nodes"]) or graph_fingerprint(graph) != state["graph_sha256"]:
            raise ValueError("checkpoint was trained on a different graph or node-ID mapping")
        self.model = SPAGANGBC(**state["model_config"]).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.x = degree_features(graph.edge_index, graph.num_nodes).to(self.device)
        self.paths = sample_shortest_paths(graph.edge_index, graph.num_nodes,
                                           **state["path_config"]).to(self.device)
        with torch.no_grad():
            self.h = self.model.encode(self.x, self.paths)

    @torch.no_grad()
    def predict_batch(self, groups: Sequence[Iterable[int]], batch_size: int = 256) -> list[float]:
        if not groups:
            return []
        canonical = [sorted(map(int, group)) for group in groups]
        if any(len(group) != self.k or len(set(group)) != self.k for group in canonical):
            raise ValueError(f"this checkpoint scores only full k={self.k} sets")
        out: list[float] = []
        for start in range(0, len(canonical), batch_size):
            ids = torch.tensor(canonical[start:start + batch_size], dtype=torch.long,
                               device=self.device)
            scores = self.model.score_encoded(self.h, ids)
            out.extend(scores.detach().cpu().tolist())
        return out

    def score(self, group: Iterable[int]) -> float:
        """Return predicted unnormalized GBC for one full k-vertex group."""
        return self.predict_batch([group])[0]

# ═══════════════════════════════════════════════════════════════════════════════
#  Main Phase 0 runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_phase0(
    graph: GraphData,
    exact_cpp_source: str | Path,
    output_dir: str | Path,
    config: Phase0Config | None = None,
    device: str | torch.device | None = None,
) -> dict:
    """Run the complete pretraining pipeline on an already loaded graph.

    Parameters
    ----------
    graph : GraphData
        Undirected, unweighted graph; vertices use IDs 0 to N-1.
    exact_cpp_source : str | Path
        The user's standalone exact_gbc.cpp file, compiled automatically.
    output_dir : str | Path
        Directory for labels, best checkpoint and validation report.
    config : Phase0Config | None
        Defaults to 1,000 unique k=10 sets, split into 800 train / 200 val.
    device : str | torch.device | None
        Default: CUDA when available, otherwise CPU.

    Returns
    -------
    dict
        Label and checkpoint paths, validation metrics and sample counts.
    """
    cfg = config or Phase0Config()
    if cfg.k != cfg.training.k or cfg.seed != cfg.training.seed:
        raise ValueError("data k/seed and training k/seed must match")
    if cfg.num_groups < 2 or not 0 < cfg.train_fraction < 1:
        raise ValueError("need >=2 groups and a train fraction in (0,1)")
    if cfg.training.max_epochs < 1 or cfg.training.batch_size < 1:
        raise ValueError("max_epochs and batch_size must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # --- Prepare the exact scorer ---
    binary = compile_exact_gbc(exact_cpp_source, output / "exact_gbc_internal")

    # --- Generate and cache exact labels ---
    print(f"[Phase0] Labeling {cfg.num_groups} random groups of k={cfg.k} vertices")
    labels = create_exact_dataset(graph, binary, output / "labels", k=cfg.k,
                                  count=cfg.num_groups, seed=cfg.seed,
                                  exact_batch_size=cfg.exact_batch_size,
                                  threads=cfg.exact_threads)

    # --- Train and evaluate on held-out groups ---
    chosen = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Phase0] Training SPAGAN-GBC on {chosen}")
    report = train_spagan_gbc(graph, labels, output / "spagan_gbc_best.pt",
                              config=cfg.training, device=chosen,
                              train_fraction=cfg.train_fraction)

    # --- Save the summary for later phases ---
    report["labels"] = str(labels)
    report["device"] = str(chosen)
    report_path = output / "phase0_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[Phase0] Best epoch={report['best_epoch']}, "
          f"validation Huber={report['validation_huber_raw']:.4f}")
    print(f"[Phase0] Saved checkpoint: {report['checkpoint']}")
    return report

# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════════════


def _smoke_test(exact_cpp_source: str | Path) -> None:
    """Check exact labels, model training, checkpoint, and group-order invariance."""
    import tempfile
    from types import SimpleNamespace

    # A path graph with one orientation of each undirected edge.
    n = 15
    edges = torch.tensor([(u, u + 1) for u in range(n - 1)],
                         dtype=torch.long).t().contiguous()
    graph = (GraphData(edge_index=edges, num_nodes=n)
             if GraphData is not Any
             else SimpleNamespace(edge_index=edges, num_nodes=n, edge_weight=None))
    training = SPAGANTrainConfig(k=10, max_epochs=2, patience=2, batch_size=8,
                                 hidden_dim=32, n_layers=2, n_heads=4)
    config = Phase0Config(k=10, num_groups=20, exact_batch_size=10,
                          training=training)
    with tempfile.TemporaryDirectory() as tmp:
        report = run_phase0(graph, exact_cpp_source, tmp, config=config, device="cpu")
        assert (report["train_count"], report["validation_count"]) == (16, 4)
        scorer = SPAGANScorer(graph, report["checkpoint"])
        score = scorer.score(range(10))
        assert math.isfinite(score) and score >= 0
        assert abs(score - scorer.score(reversed(range(10)))) < 1e-4
        assert math.isfinite(report["validation_huber_raw"])

        # The main smoke train is small; also exercise the paper-sized encoder.
        paper_model = SPAGANGBC().eval()
        x = degree_features(graph.edge_index, graph.num_nodes)
        paths = sample_shortest_paths(graph.edge_index, graph.num_nodes)
        with torch.no_grad():
            paper_score = paper_model(x, paths, torch.arange(10)[None, :])
        assert paper_score.shape == (1,) and bool(torch.isfinite(paper_score).all())
        # Independent exact check for the graph's unique shortest paths.
        labels = Path(report["labels"]).read_text(encoding="utf-8").splitlines()
        assert len(labels) == 20
        for line in labels:
            record = json.loads(line)
            C = set(record["nodes"])
            expected = sum(bool(C.intersection(range(min(s, t) + 1,
                                                    max(s, t))))
                           for s in range(n) for t in range(n) if s != t)
            assert abs(record["raw_gbc"] - expected) < 1e-7
    print("[Phase0] smoke test: PASS")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPAGAN-GBC Phase 0 smoke test")
    parser.add_argument("--exact-cpp", type=Path, default=Path("exact_gbc.cpp"),
                        help="exact C++ source (default: ./exact_gbc.cpp)")
    args = parser.parse_args()
    _smoke_test(args.exact_cpp)
