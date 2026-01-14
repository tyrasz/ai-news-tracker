#!/usr/bin/env python3
"""Database migration script to add new columns for multi-user support."""

import sqlite3
import sys


def migrate(db_path: str = "news_tracker.db"):
    """Add missing columns to existing database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    migrations = [
        # Articles table - bookmark columns
        ("articles", "is_bookmarked", "BOOLEAN DEFAULT 0"),
        ("articles", "bookmarked_at", "DATETIME"),

        # Article notes - user_id column
        ("article_notes", "user_id", "INTEGER REFERENCES users(id)"),

        # User profile - user_id column
        ("user_profile", "user_id", "INTEGER REFERENCES users(id)"),
    ]

    # New tables
    new_tables = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email VARCHAR(256) UNIQUE NOT NULL,
            password_hash VARCHAR(256) NOT NULL,
            api_key VARCHAR(64) UNIQUE NOT NULL,
            display_name VARCHAR(256),
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME,
            last_login DATETIME
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_article_interactions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            article_id INTEGER NOT NULL REFERENCES articles(id),
            is_read BOOLEAN DEFAULT 0,
            is_liked BOOLEAN,
            is_bookmarked BOOLEAN DEFAULT 0,
            read_at DATETIME,
            bookmarked_at DATETIME,
            liked_at DATETIME
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_user_article
        ON user_article_interactions(user_id, article_id)
        """,
    ]

    print(f"Migrating database: {db_path}")

    # Create new tables
    for sql in new_tables:
        try:
            cursor.execute(sql)
            print(f"  Created table/index")
        except sqlite3.OperationalError as e:
            if "already exists" in str(e):
                print(f"  Table/index already exists, skipping")
            else:
                print(f"  Warning: {e}")

    # Add missing columns
    for table, column, definition in migrations:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            print(f"  Added {table}.{column}")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print(f"  Column {table}.{column} already exists, skipping")
            else:
                print(f"  Warning: {e}")

    conn.commit()
    conn.close()
    print("Migration complete!")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "news_tracker.db"
    migrate(db_path)
