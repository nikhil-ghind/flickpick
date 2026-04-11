"""
Metrics computation for the analytics dashboard.

Computes recommendation quality metrics:
- Diversity: intra-list diversity (genre spread)
- Novelty: how often users see items outside their typical preferences
- Coverage: what fraction of the catalog gets recommended
- Serendipity: surprisingly good recommendations
"""

import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


def intra_list_diversity(recommended_items: list[dict], genre_key: str = "genre_ids") -> float:
    """Compute genre diversity within a recommendation list.

    Returns the average pairwise dissimilarity (1 - Jaccard) across all item pairs.
    Higher = more diverse recommendations.
    """
    if len(recommended_items) < 2:
        return 0.0

    genres_list = [set(item.get(genre_key, [])) for item in recommended_items]
    total_dissimilarity = 0.0
    n_pairs = 0

    for i in range(len(genres_list)):
        for j in range(i + 1, len(genres_list)):
            a, b = genres_list[i], genres_list[j]
            if not a and not b:
                dissimilarity = 0.0
            else:
                intersection = len(a & b)
                union = len(a | b)
                dissimilarity = 1.0 - (intersection / union) if union > 0 else 0.0
            total_dissimilarity += dissimilarity
            n_pairs += 1

    return total_dissimilarity / n_pairs if n_pairs > 0 else 0.0


def catalog_coverage(
    all_recommendations: list[list[str]],
    total_items: int,
) -> float:
    """What fraction of the catalog appears in any user's recommendations.

    Low coverage = popularity bias (only popular items get recommended).
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def novelty_score(
    recommended_item_ids: list[str],
    item_popularity: dict[str, float],
) -> float:
    """Average self-information of recommended items.

    Items with lower popularity have higher novelty.
    novelty(item) = -log2(P(item))
    """
    if not recommended_item_ids:
        return 0.0

    total_pop = sum(item_popularity.values())
    if total_pop == 0:
        return 0.0

    novelties = []
    for item_id in recommended_item_ids:
        pop = item_popularity.get(item_id, 1)
        prob = pop / total_pop
        novelties.append(-np.log2(prob + 1e-10))

    return float(np.mean(novelties))


def genre_distribution(recommended_items: list[dict], genre_key: str = "genre_ids") -> dict[int, float]:
    """Compute the genre distribution across recommendations.

    Returns: {genre_id: fraction}
    """
    counter: Counter = Counter()
    total = 0
    for item in recommended_items:
        for g in item.get(genre_key, []):
            counter[g] += 1
            total += 1

    if total == 0:
        return {}
    return {g: count / total for g, count in counter.most_common()}


def recommendation_quality_report(
    recommendations: dict[str, list[dict]],
    item_popularity: dict[str, float],
    total_items: int,
) -> dict:
    """Generate a comprehensive recommendation quality report.

    Args:
        recommendations: {user_id: [list of recommended item dicts]}
        item_popularity: {item_id: view_count}
        total_items: total number of items in catalog

    Returns:
        Dictionary of aggregate metrics
    """
    diversities = []
    novelties = []
    all_rec_ids = []

    for user_id, recs in recommendations.items():
        diversities.append(intra_list_diversity(recs))
        rec_ids = [r.get("item_id", "") for r in recs]
        novelties.append(novelty_score(rec_ids, item_popularity))
        all_rec_ids.append(rec_ids)

    return {
        "avg_diversity": float(np.mean(diversities)) if diversities else 0.0,
        "avg_novelty": float(np.mean(novelties)) if novelties else 0.0,
        "catalog_coverage": catalog_coverage(all_rec_ids, total_items),
        "num_users_evaluated": len(recommendations),
    }
