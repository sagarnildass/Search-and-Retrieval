"""
End-to-End Demo Script for FAISS Retrieval + Query Expansion + Cross-Encoder Re-ranking

What this adds vs your current script:
1) Query expansion (open-source) using a lightweight HF generator by default (Flan-T5),
   with a clean hook to swap to LLaMA (via Transformers or llama.cpp/Ollama).
2) Pretrained CrossEncoder re-ranker (Sentence-Transformers) to re-rank top-N FAISS candidates.

Key idea:
- FAISS (HNSW/Flat) gives fast candidate retrieval
- (Optional) Query Expansion improves recall
- Cross-Encoder re-ranking improves precision/ordering

Dependencies:
  pip install faiss-cpu sentence-transformers pandas pyarrow transformers torch
Optional (if you want LLaMA locally):
  pip install llama-cpp-python
or use Ollama installed on your machine (no pip needed for Python, you’d call its HTTP API)

Files expected in data/ (same as your current):
  metadata.pkl
  faiss_index_flat.bin
  faiss_index_hnsw.bin
  product_ids.pkl
  products_clean.parquet
"""

import os
import time
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).parent.parent
INDEX_DIR = Path(__file__).parent / "data"


# -----------------------------
# Config
# -----------------------------
@dataclass
class RerankConfig:
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # retrieve k candidates from FAISS, rerank top_n among them
    top_n: int = 50  # must be <= k
    batch_size: int = 64


@dataclass
class ExpansionConfig:
    enabled: bool = True
    # "hf_flan_t5" works well and is small. Swap to a LLaMA-based expander if you want.
    mode: str = "hf_flan_t5"  # options: "hf_flan_t5", "none"
    model_name: str = "google/flan-t5-base"
    num_expansions: int = 4
    max_new_tokens: int = 96
    temperature: float = 0.7


@dataclass
class SearchConfig:
    default_k: int = 20
    candidate_k: int = 100  # how many to pull from FAISS before rerank (if enabled)
    index_type: str = "hnsw"  # "flat" or "hnsw"
    hnsw_default_ef_search: int = 200


# -----------------------------
# Query Expansion (HF Flan-T5)
# -----------------------------
class QueryExpander:
    """
    Generates short query variants/synonyms to expand recall.

    Default implementation uses a small open-source instruction model (Flan-T5).
    If you want LLaMA, keep the interface and swap implementation.
    """

    def __init__(self, cfg: ExpansionConfig):
        self.cfg = cfg
        self._pipe = None

        if self.cfg.enabled and self.cfg.mode not in ["hf_flan_t5", "none"]:
            raise ValueError(f"Unknown expansion mode: {self.cfg.mode}")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def expand(self, query: str) -> List[str]:
        """
        Returns a list of expanded queries (not including the original).
        """
        if not self.cfg.enabled or self.cfg.mode == "none":
            return []

        if self.cfg.mode == "hf_flan_t5":
            if self._pipe is None:
                from transformers import pipeline

                self._pipe = pipeline(
                    "text2text-generation",
                    model=self.cfg.model_name,
                    device=0 if self._has_cuda() else -1,
                )

            prompt = (
                "You are a search assistant for an e-commerce product catalog.\n"
                "Task: expand a short user query into multiple alternative search queries.\n"
                "Guidelines:\n"
                "- Preserve the original intent and product category.\n"
                "- Use natural shopping language and common synonyms.\n"
                "- Add helpful attributes only if they are typical for the item (e.g., size, color, material),\n"
                "  but do not invent specifics that are not implied.\n"
                "- Keep each query concise (4-10 words).\n"
                "- Do NOT add brand names unless mentioned in the original query.\n"
                "- Return exactly {n} alternatives.\n"
                "- Output each query on its own line with no numbering or bullets.\n"
                "\n"
                "Original query: {q}\n"
                "Expanded queries:\n"
            ).format(n=self.cfg.num_expansions, q=query)

            out = self._pipe(
                prompt,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                num_return_sequences=1,
            )[0]["generated_text"]

            # Parse lines, clean, dedupe
            lines = [l.strip(" -•\t").strip() for l in out.splitlines()]
            lines = [l for l in lines if l and l.lower() != query.lower()]

            # Heuristic: sometimes model returns numbered list in one line
            if len(lines) <= 1 and ("\n" not in out):
                # split by common separators
                for sep in ["; ", " | ", " . ", ", "]:
                    if sep in out:
                        parts = [p.strip() for p in out.split(sep)]
                        parts = [p for p in parts if p and p.lower() != query.lower()]
                        if len(parts) > len(lines):
                            lines = parts
                            break

            # Dedupe preserving order
            seen = set()
            expansions = []
            for l in lines:
                key = l.lower()
                if key not in seen:
                    seen.add(key)
                    expansions.append(l)

            return expansions[: self.cfg.num_expansions]

        return []


# -----------------------------
# Retrieval System
# -----------------------------
class RetrievalDemoQE_Rerank:
    def __init__(
        self,
        index_dir=INDEX_DIR,
        search_cfg: SearchConfig = SearchConfig(),
        exp_cfg: ExpansionConfig = ExpansionConfig(),
        rerank_cfg: RerankConfig = RerankConfig(),
    ):
        self.index_dir = Path(index_dir)

        self.search_cfg = search_cfg
        self.exp_cfg = exp_cfg
        self.rerank_cfg = rerank_cfg

        self.flat_index = None
        self.hnsw_index = None
        self.embedding_model = None
        self.cross_encoder = None
        self.expander = None

        self.product_ids = None
        self.products_df = None
        self.embedding_dim = None

        # Fast maps
        self._id_to_title: Dict = {}

    def load(self):
        print("=" * 80)
        print("LOADING RETRIEVAL SYSTEM (QE + RERANK)")
        print("=" * 80)

        # Load metadata
        print("\n1. Loading metadata...")
        with open(self.index_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.embedding_dim = metadata["embedding_dim"]
        model_name = metadata["model_name"]
        print(f"   Embedding dimension: {self.embedding_dim}")
        print(f"   Embedding model: {model_name}")
        print(f"   Number of products: {metadata['n_products']:,}")

        # Load embedding model
        print("\n2. Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"   Loaded: {model_name}")
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        # Load FAISS indices
        print("\n3. Loading FAISS indices...")
        self.flat_index = faiss.read_index(str(self.index_dir / "faiss_index_flat.bin"))
        print(f"   Flat index loaded: {self.flat_index.ntotal:,} vectors")

        self.hnsw_index = faiss.read_index(str(self.index_dir / "faiss_index_hnsw.bin"))
        self.hnsw_index.hnsw.efSearch = self.search_cfg.hnsw_default_ef_search
        print(f"   HNSW index loaded: {self.hnsw_index.ntotal:,} vectors")
        print(f"   HNSW ef_search set to: {self.hnsw_index.hnsw.efSearch}")
        faiss.omp_set_num_threads(1)

        # Load product IDs
        print("\n4. Loading product IDs...")
        with open(self.index_dir / "product_ids.pkl", "rb") as f:
            self.product_ids = pickle.load(f)
        print(f"   Loaded {len(self.product_ids):,} product IDs")

        # Load products dataframe
        print("\n5. Loading products dataframe...")
        self.products_df = pd.read_parquet(self.index_dir / "products_clean.parquet")
        print(f"   Loaded {len(self.products_df):,} products")

        # Build fast maps (avoid per-result dataframe filtering)
        print("\n6. Building fast lookup maps...")
        # Expect columns: product_id, product_title
        if "product_id" not in self.products_df.columns or "product_title" not in self.products_df.columns:
            raise ValueError("products_clean.parquet must have columns: product_id, product_title")
        self._id_to_title = dict(
            zip(self.products_df["product_id"].tolist(), self.products_df["product_title"].tolist())
        )
        print("   Built product_id -> title map")

        # Query expander
        print("\n7. Initializing query expander...")
        self.expander = QueryExpander(self.exp_cfg)
        print(f"   QE enabled: {self.exp_cfg.enabled} (mode={self.exp_cfg.mode})")

        # Cross-encoder reranker
        print("\n8. Loading cross-encoder reranker...")
        if self.rerank_cfg.enabled:
            print(f"   Reranker will load on first use: {self.rerank_cfg.model_name}")
        else:
            print("   Reranker disabled")

        print("\nSystem loaded successfully!")

    # -----------------------------
    # Core search pieces
    # -----------------------------
    def _embed_queries(self, queries: List[str]) -> np.ndarray:
        embs = self.embedding_model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return embs

    def _faiss_search(
        self,
        query_emb: np.ndarray,
        k: int,
        index_type: str,
        ef_search: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if index_type not in ["flat", "hnsw"]:
            raise ValueError("index_type must be 'flat' or 'hnsw'")

        index = self.flat_index if index_type == "flat" else self.hnsw_index

        original_ef = None
        if index_type == "hnsw" and ef_search is not None:
            original_ef = index.hnsw.efSearch
            index.hnsw.efSearch = ef_search

        start = time.time()
        distances, indices = index.search(query_emb.reshape(1, -1), k)
        ms = (time.time() - start) * 1000.0

        if original_ef is not None:
            index.hnsw.efSearch = original_ef

        return distances[0], indices[0], ms

    def _collect_candidates(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
    ) -> List[dict]:
        """
        Turn FAISS hits into candidate dicts.
        """
        cands = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.product_ids):
                continue
            pid = self.product_ids[idx]
            title = self._id_to_title.get(pid)
            if title is None:
                continue
            cands.append(
                {
                    "product_id": pid,
                    "title": title,
                    "faiss_distance": float(dist),
                }
            )
        return cands

    def _dedupe_candidates(self, cands: List[dict]) -> List[dict]:
        seen = set()
        out = []
        for c in cands:
            pid = c["product_id"]
            if pid not in seen:
                seen.add(pid)
                out.append(c)
        return out

    def _rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int,
    ) -> List[dict]:
        """
        Cross-encoder rerank: score (query, title) pairs and sort by score desc.
        """
        if not self.rerank_cfg.enabled:
            return candidates[:top_k]

        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder(self.rerank_cfg.model_name)

        rerank_n = min(self.rerank_cfg.top_n, len(candidates))
        rerank_n = min(rerank_n, len(candidates))

        pairs = [(query, candidates[i]["title"]) for i in range(rerank_n)]
        start = time.time()
        scores = self.cross_encoder.predict(pairs, batch_size=self.rerank_cfg.batch_size)
        rerank_ms = (time.time() - start) * 1000.0

        for i in range(rerank_n):
            candidates[i]["rerank_score"] = float(scores[i])
            candidates[i]["rerank_ms"] = rerank_ms

        # For the rest, set score to None so they naturally fall after reranked ones
        for i in range(rerank_n, len(candidates)):
            candidates[i]["rerank_score"] = None
            candidates[i]["rerank_ms"] = rerank_ms

        # Sort: rerank_score desc first; fallback to faiss_distance asc
        candidates_sorted = sorted(
            candidates,
            key=lambda c: (
                -(c["rerank_score"] if c["rerank_score"] is not None else -1e9),
                c["faiss_distance"],
            ),
        )
        return candidates_sorted[:top_k]

    # -----------------------------
    # Public search API
    # -----------------------------
    def search(
        self,
        query: str,
        k: int = None,
        index_type: str = None,
        ef_search: Optional[int] = None,
        use_expansion: Optional[bool] = None,
        use_rerank: Optional[bool] = None,
    ) -> dict:
        """
        Returns a dict with:
          - final_results: list of result dicts
          - debug: timings and expansions
        """
        if k is None:
            k = self.search_cfg.default_k
        if index_type is None:
            index_type = self.search_cfg.index_type

        if use_expansion is None:
            use_expansion = self.exp_cfg.enabled
        if use_rerank is None:
            use_rerank = self.rerank_cfg.enabled

        # Candidate pool size for rerank
        candidate_k = max(self.search_cfg.candidate_k, k)
        if use_rerank:
            candidate_k = max(candidate_k, self.rerank_cfg.top_n, k)

        debug = {
            "query": query,
            "index_type": index_type,
            "k": k,
            "candidate_k": candidate_k,
            "ef_search": None,
            "expansions": [],
            "timings_ms": {},
        }

        if index_type == "hnsw":
            debug["ef_search"] = self.hnsw_index.hnsw.efSearch if ef_search is None else ef_search

        # 1) Expand queries (optional)
        expansions = []
        t0 = time.time()
        if use_expansion and self.expander is not None:
            expansions = self.expander.expand(query)
        debug["timings_ms"]["qe"] = (time.time() - t0) * 1000.0
        debug["expansions"] = expansions

        all_queries = [query] + expansions

        # 2) Embed all queries
        t0 = time.time()
        all_embs = self._embed_queries(all_queries)
        debug["timings_ms"]["embed"] = (time.time() - t0) * 1000.0

        # 3) FAISS search for each query, merge candidates
        t0 = time.time()
        merged = []
        faiss_times = []
        for qi, q_emb in zip(all_queries, all_embs):
            dists, idxs, ms = self._faiss_search(q_emb, candidate_k, index_type=index_type, ef_search=ef_search)
            faiss_times.append(ms)
            cands = self._collect_candidates(idxs, dists)
            # Keep info about which query produced it (useful for debugging)
            for c in cands:
                c["matched_query"] = qi
            merged.extend(cands)

        debug["timings_ms"]["faiss_total"] = (time.time() - t0) * 1000.0
        debug["timings_ms"]["faiss_avg_per_query"] = float(np.mean(faiss_times)) if faiss_times else 0.0

        # Dedupe by product_id while preserving earlier order (usually better FAISS rank first)
        merged = self._dedupe_candidates(merged)

        # 4) Rerank (optional) using ORIGINAL query (not expanded strings)
        # Temporarily flip rerank flag based on use_rerank
        original_rerank_enabled = self.rerank_cfg.enabled
        self.rerank_cfg.enabled = bool(use_rerank)

        t0 = time.time()
        final = self._rerank(query, merged, top_k=k)
        debug["timings_ms"]["rerank"] = (time.time() - t0) * 1000.0 if use_rerank else 0.0

        # Restore config
        self.rerank_cfg.enabled = original_rerank_enabled

        # Attach simple output rows
        final_results = []
        for rank, c in enumerate(final, 1):
            final_results.append(
                {
                    "rank": rank,
                    "product_id": c["product_id"],
                    "title": c["title"],
                    "faiss_distance": c["faiss_distance"],
                    "rerank_score": c.get("rerank_score", None),
                    "matched_query": c.get("matched_query", query),
                }
            )

        return {"final_results": final_results, "debug": debug}

    # -----------------------------
    # Display helpers
    # -----------------------------
    def display_results(self, result_obj: dict):
        debug = result_obj["debug"]
        results = result_obj["final_results"]

        print("\n" + "=" * 80)
        print(f"QUERY: '{debug['query']}'")
        print(
            f"Index: {debug['index_type'].upper()} | "
            f"K={debug['k']} | CandidateK={debug['candidate_k']} | "
            f"QE={'ON' if (self.exp_cfg.enabled) else 'OFF'} | "
            f"RERANK={'ON' if (self.rerank_cfg.enabled) else 'OFF'}"
        )
        if debug["index_type"] == "hnsw":
            print(f"HNSW ef_search: {debug['ef_search']}")
        print("=" * 80)

        if debug["expansions"]:
            print("\nQuery expansions:")
            for i, e in enumerate(debug["expansions"], 1):
                print(f"  {i}. {e}")
        elif self.exp_cfg.enabled:
            print("\nQuery expansions: (none)")

        if self.exp_cfg.enabled:
            all_queries = [debug["query"]] + debug["expansions"]
            print("\nAll queries used:")
            for i, q in enumerate(all_queries, 1):
                print(f"  {i}. {q}")

        t = debug["timings_ms"]
        print("\nTimings (ms):")
        print(f"  QE:           {t.get('qe', 0.0):.2f}")
        print(f"  Embed:        {t.get('embed', 0.0):.2f}")
        print(f"  FAISS total:  {t.get('faiss_total', 0.0):.2f} (avg/query {t.get('faiss_avg_per_query', 0.0):.2f})")
        if self.rerank_cfg.enabled:
            print(f"  Rerank step:  {t.get('rerank', 0.0):.2f}")

        if not results:
            print("\nNo results found.")
            return

        print("\nResults:")
        print("-" * 80)
        for r in results:
            print(f"\n{r['rank']}. [{r['product_id']}]")
            title = r["title"]
            print(f"   Title: {title[:110]}{'...' if len(title) > 110 else ''}")
            print(f"   FAISS distance: {r['faiss_distance']:.4f}")
            if r["rerank_score"] is not None:
                print(f"   Rerank score:   {r['rerank_score']:.4f}")
            if r["matched_query"] != debug["query"]:
                print(f"   Matched via:    {r['matched_query']}")

    def compare_indices(
        self,
        query: str,
        k: int = 20,
        ef_search: Optional[int] = None,
        use_expansion: Optional[bool] = None,
        use_rerank: Optional[bool] = None,
    ):
        print("\n" + "=" * 80)
        print(f"COMPARING INDICES FOR QUERY: '{query}'")
        print("=" * 80)

        flat_obj = self.search(
            query=query,
            k=k,
            index_type="flat",
            ef_search=None,
            use_expansion=use_expansion,
            use_rerank=use_rerank,
        )
        hnsw_obj = self.search(
            query=query,
            k=k,
            index_type="hnsw",
            ef_search=ef_search,
            use_expansion=use_expansion,
            use_rerank=use_rerank,
        )

        print("\n--- FLAT ---")
        self.display_results(flat_obj)

        print("\n--- HNSW ---")
        self.display_results(hnsw_obj)

        flat_ids = [r["product_id"] for r in flat_obj["final_results"]]
        hnsw_ids = [r["product_id"] for r in hnsw_obj["final_results"]]
        overlap = set(flat_ids).intersection(set(hnsw_ids))

        # Timings: use FAISS total as a proxy; (includes multiple queries if QE)
        ft = flat_obj["debug"]["timings_ms"].get("faiss_total", 0.0)
        ht = hnsw_obj["debug"]["timings_ms"].get("faiss_total", 0.0)

        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print("\nFAISS stage speed (ms):")
        print(f"  Flat: {ft:.2f}")
        print(f"  HNSW: {ht:.2f}")
        if ht > 0:
            print(f"  Speedup (FAISS stage): {ft/ht:.2f}x")

        print("\nResult overlap:")
        print(f"  Common: {len(overlap)}/{k} ({(len(overlap)/k*100):.1f}%)")


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 80)
    print("FAISS RETRIEVAL DEMO (QUERY EXPANSION + CROSS-ENCODER RERANK)")
    print("=" * 80)

    demo = RetrievalDemoQE_Rerank()
    demo.load()

    example_queries = [
        "wireless bluetooth headphones",
        "laptop computer",
        "running shoes",
        "coffee maker",
        "smartphone case",
    ]

    print("\n" + "=" * 80)
    print("DEMO: SINGLE INDEX SEARCH (HNSW default)")
    print("=" * 80)

    for q in example_queries[:3]:
        obj = demo.search(q, k=10, index_type="hnsw")
        demo.display_results(obj)

    print("\n" + "=" * 80)
    print("DEMO: COMPARING BOTH INDICES")
    print("=" * 80)

    demo.compare_indices(example_queries[0], k=20)

    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)

    print("\nType queries (or 'quit' to exit).")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not query:
                continue

            print("\nChoose index:")
            print("  1. Flat (exact)")
            print("  2. HNSW (approximate)")
            print("  3. Compare both")
            print("  4. HNSW with custom K")
            print("  5. Toggle Query Expansion (currently: {})".format("ON" if demo.exp_cfg.enabled else "OFF"))
            print("  6. Toggle Re-ranking (currently: {})".format("ON" if demo.rerank_cfg.enabled else "OFF"))
            print("  7. Set HNSW ef_search (currently: {})".format(demo.hnsw_index.hnsw.efSearch))

            choice = input("Choice (1-7): ").strip()

            if choice == "5":
                demo.exp_cfg.enabled = not demo.exp_cfg.enabled
                demo.expander = QueryExpander(demo.exp_cfg)
                print(f"Query Expansion is now: {'ON' if demo.exp_cfg.enabled else 'OFF'}")
                continue

            if choice == "6":
                demo.rerank_cfg.enabled = not demo.rerank_cfg.enabled
                if demo.rerank_cfg.enabled and demo.cross_encoder is None:
                    demo.cross_encoder = CrossEncoder(demo.rerank_cfg.model_name)
                print(f"Re-ranking is now: {'ON' if demo.rerank_cfg.enabled else 'OFF'}")
                continue

            if choice == "7":
                try:
                    val = int(input("Enter ef_search (e.g., 50/100/200/400): ").strip())
                    demo.hnsw_index.hnsw.efSearch = val
                    print(f"HNSW ef_search set to: {demo.hnsw_index.hnsw.efSearch}")
                except ValueError:
                    print("Invalid ef_search.")
                continue

            if choice == "1":
                obj = demo.search(query, k=20, index_type="flat")
                demo.display_results(obj)

            elif choice == "2":
                obj = demo.search(query, k=20, index_type="hnsw")
                demo.display_results(obj)

            elif choice == "3":
                demo.compare_indices(query, k=20)

            elif choice == "4":
                try:
                    k_val = int(input("Enter K (e.g., 50, 100, 200): ").strip())
                    obj = demo.search(query, k=k_val, index_type="hnsw")
                    demo.display_results(obj)
                except ValueError:
                    print("Invalid number, using K=20.")
                    obj = demo.search(query, k=20, index_type="hnsw")
                    demo.display_results(obj)

            else:
                print("Invalid choice. Using HNSW K=20.")
                obj = demo.search(query, k=20, index_type="hnsw")
                demo.display_results(obj)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
