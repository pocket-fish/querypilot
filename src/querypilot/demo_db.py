import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    amount REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_COUNTRIES = ["Germany", "France", "Spain", "Poland", "Italy"]
_STATUSES = ["paid", "paid", "paid", "pending", "cancelled"]


def create_demo_db(path: str, seed: int = 42, customers: int = 30, orders: int = 300) -> None:
    """Create a deterministic demo database (overwrites an existing file)."""
    rng = random.Random(seed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    start = date(2025, 1, 1)
    with sqlite3.connect(target) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [(i, f"Customer {i:02d}", rng.choice(_COUNTRIES)) for i in range(1, customers + 1)],
        )
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            [
                (
                    i,
                    rng.randint(1, customers),
                    round(rng.uniform(10, 500), 2),
                    rng.choice(_STATUSES),
                    (start + timedelta(days=rng.randint(0, 364))).isoformat(),
                )
                for i in range(1, orders + 1)
            ],
        )
