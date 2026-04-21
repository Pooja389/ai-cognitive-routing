"""
Phase 1: Vector-Based Persona Matching (The Router)
Uses FAISS + sentence-transformers to embed bot personas and route
incoming posts to relevant bots via cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bot Personas
# ---------------------------------------------------------------------------

BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "persona": (
            "I believe AI and crypto will solve all human problems. I am highly optimistic "
            "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "persona": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "persona": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}

# ---------------------------------------------------------------------------
# VectorStore — thin wrapper around FAISS
# ---------------------------------------------------------------------------

class PersonaVectorStore:
    """
    In-memory FAISS index storing one vector per bot persona.
    Uses cosine similarity (via inner-product on L2-normalised vectors).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        # IndexFlatIP  →  dot-product on unit vectors  ==  cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.bot_ids: list[str] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed and L2-normalise a list of strings."""
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(vecs)
        return vecs.astype("float32")

    def add_personas(self, personas: dict) -> None:
        """Embed every persona and add to the FAISS index."""
        texts = [v["persona"] for v in personas.values()]
        ids   = list(personas.keys())
        vecs  = self._embed(texts)
        self.index.add(vecs)
        self.bot_ids.extend(ids)
        logger.info(f"Indexed {len(ids)} personas: {ids}")

    def query(self, text: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """
        Return [(bot_id, cosine_similarity)] for all stored bots,
        sorted by descending similarity.
        """
        k = top_k or len(self.bot_ids)
        vec = self._embed([text])
        scores, indices = self.index.search(vec, k)
        results = [
            (self.bot_ids[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx != -1
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Public routing function
# ---------------------------------------------------------------------------

# Module-level store (lazy-initialised once)
_store: PersonaVectorStore | None = None


def _get_store() -> PersonaVectorStore:
    global _store
    if _store is None:
        _store = PersonaVectorStore()
        _store.add_personas(BOT_PERSONAS)
    return _store


def route_post_to_bots(post_content: str, threshold: float = 0.15) -> list[dict]:
    """
    Embed *post_content* and return metadata for every bot whose persona
    cosine-similarity exceeds *threshold*.

    Note on threshold: all-MiniLM-L6-v2 scores typically land in [0.25, 0.65]
    for topically related but not identical texts, so the default is 0.40.
    Pass threshold=0.85 only with models trained for high-precision retrieval.

    Returns:
        List of dicts: [{"bot_id": ..., "name": ..., "similarity": ...}]
    """
    store   = _get_store()
    results = store.query(post_content)

    matched = []
    logger.info(f'\nRouting post: "{post_content[:80]}..."')
    logger.info(f"{'Bot':<10} {'Similarity':>12}")
    logger.info("-" * 25)
    for bot_id, sim in results:
        sim_pct = round(sim * 100, 1)
        marker = "✓ MATCHED" if sim >= threshold else "✗ below threshold"
        logger.info(f"{bot_id:<10} {sim_pct:>10.1f}%   {marker}")
        if sim >= threshold:
            matched.append({
        "bot_id":     bot_id,
        "name":       BOT_PERSONAS[bot_id]["name"],
        "similarity": f"{sim_pct}%",
    })

    logger.info(f"\n→ {len(matched)} bot(s) matched above threshold {threshold}.")
    return matched


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "The Fed raised interest rates again — bond yields are spiking.",
        "Big Tech is buying up farmland and displacing rural communities.",
    ]

    for post in test_posts:
        print("\n" + "=" * 60)
        matches = route_post_to_bots(post)
        print(f"Matched bots: {[m['bot_id'] for m in matches]}")
