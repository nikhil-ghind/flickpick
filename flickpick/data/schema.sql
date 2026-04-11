-- Flickpick database schema

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS items (
    item_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    duration_minutes FLOAT,
    release_date DATE,
    avg_rating FLOAT DEFAULT 0.0,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS genres (
    genre_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS item_genres (
    item_id INTEGER REFERENCES items(item_id),
    genre_id INTEGER REFERENCES genres(genre_id),
    PRIMARY KEY (item_id, genre_id)
);

CREATE TABLE IF NOT EXISTS watch_events (
    event_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    item_id INTEGER REFERENCES items(item_id),
    watch_pct FLOAT NOT NULL,           -- 0.0 to 1.0
    watch_seconds INTEGER NOT NULL,
    session_id VARCHAR(64),
    device_type VARCHAR(20),
    watched_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_watch_events_user ON watch_events(user_id, watched_at DESC);
CREATE INDEX idx_watch_events_item ON watch_events(item_id, watched_at DESC);
CREATE INDEX idx_watch_events_time ON watch_events(watched_at);

CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER REFERENCES users(user_id),
    item_id INTEGER REFERENCES items(item_id),
    rating FLOAT NOT NULL,
    rated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, item_id)
);

CREATE TABLE IF NOT EXISTS experiment_assignments (
    user_id INTEGER REFERENCES users(user_id),
    experiment_id VARCHAR(100) NOT NULL,
    variant VARCHAR(50) NOT NULL,
    assigned_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_events (
    event_id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(100) NOT NULL,
    user_id INTEGER REFERENCES users(user_id),
    variant VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_exp_events_lookup ON experiment_events(experiment_id, metric_name, variant);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id VARCHAR(50) PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,     -- 'two_tower' or 'ranker'
    artifact_path VARCHAR(500) NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);
