"""
Synthetic data generator for development and testing.

Generates realistic-looking users, items, and interaction events
with genre preferences and temporal patterns.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


GENRES = [
    "Action", "Comedy", "Drama", "Thriller", "Sci-Fi",
    "Romance", "Horror", "Documentary", "Animation", "Crime",
    "Fantasy", "Mystery", "Adventure", "Family", "Musical",
]


def generate_items(n_items: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic content catalog."""
    rng = np.random.default_rng(seed)

    items = []
    for i in range(n_items):
        n_genres = rng.integers(1, 4)
        genres = random.sample(range(len(GENRES)), n_genres)
        release_days_ago = int(rng.exponential(365))

        items.append({
            "item_id": i,
            "title": f"Title_{i}",
            "duration_minutes": float(rng.choice([22, 30, 45, 60, 90, 120, 150])),
            "release_date": (datetime.now() - timedelta(days=release_days_ago)).date(),
            "avg_rating": round(float(rng.normal(3.5, 0.8).clip(1, 5)), 1),
            "genre_ids": genres,
        })

    return pd.DataFrame(items)


def generate_users(n_users: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic users with genre preferences."""
    rng = np.random.default_rng(seed)

    users = []
    for i in range(n_users):
        # Each user has 2-4 preferred genres
        n_prefs = rng.integers(2, 5)
        preferred_genres = random.sample(range(len(GENRES)), n_prefs)
        genre_weights = rng.dirichlet(np.ones(n_prefs)).tolist()

        users.append({
            "user_id": i,
            "username": f"user_{i}",
            "preferred_genres": preferred_genres,
            "genre_weights": genre_weights,
            "activity_level": float(rng.choice(["low", "medium", "high"],
                                               p=[0.3, 0.5, 0.2] if False else [1, 1, 1])),
        })
        # Fix activity level to be a string
        users[-1]["activity_level"] = rng.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2])

    return pd.DataFrame(users)


def generate_interactions(
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    n_interactions: int = 500000,
    days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic watch events with realistic patterns.

    - Users prefer items matching their genre preferences
    - Watch percentage is higher for preferred genres
    - Temporal patterns: more activity in evenings/weekends
    """
    rng = np.random.default_rng(seed)

    interactions = []
    for _ in range(n_interactions):
        user = users_df.sample(1, random_state=rng.integers(1e9)).iloc[0]
        user_genres = set(user["preferred_genres"])

        # 70% chance of watching from preferred genre
        if rng.random() < 0.7:
            genre_items = items_df[
                items_df["genre_ids"].apply(lambda g: bool(set(g) & user_genres))
            ]
            if len(genre_items) > 0:
                item = genre_items.sample(1, random_state=rng.integers(1e9)).iloc[0]
            else:
                item = items_df.sample(1, random_state=rng.integers(1e9)).iloc[0]
        else:
            item = items_df.sample(1, random_state=rng.integers(1e9)).iloc[0]

        # Watch percentage: higher for genre matches
        genre_overlap = bool(set(item["genre_ids"]) & user_genres)
        if genre_overlap:
            watch_pct = float(rng.beta(5, 2))  # skewed toward higher %
        else:
            watch_pct = float(rng.beta(2, 3))  # skewed toward lower %

        watch_seconds = int(watch_pct * item["duration_minutes"] * 60)

        # Random timestamp in the last N days
        hours_ago = rng.exponential(days * 12)  # exponential decay
        timestamp = datetime.now() - timedelta(hours=float(hours_ago))
        hour = timestamp.hour
        device = rng.choice(["mobile", "tv", "web", "tablet"], p=[0.35, 0.30, 0.25, 0.10])

        interactions.append({
            "user_id": int(user["user_id"]),
            "item_id": int(item["item_id"]),
            "watch_pct": round(watch_pct, 3),
            "watch_seconds": watch_seconds,
            "hour": hour,
            "dow": timestamp.weekday(),
            "device": device,
            "watched_at": timestamp,
            "label": 1 if watch_pct > 0.7 else 0,  # engagement label for ranker
        })

    df = pd.DataFrame(interactions)
    return df.sort_values("watched_at").reset_index(drop=True)


def generate_dataset(
    n_users: int = 10000,
    n_items: int = 5000,
    n_interactions: int = 500000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a complete synthetic dataset."""
    users = generate_users(n_users, seed)
    items = generate_items(n_items, seed)
    interactions = generate_interactions(users, items, n_interactions, seed=seed)
    return users, items, interactions
