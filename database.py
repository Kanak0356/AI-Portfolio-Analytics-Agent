"""Database setup and CSV loader for the Portfolio Analytics Agent."""

import os
import sqlite3
import pandas as pd

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "portfolio_database.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "database_schema.sql")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Tables in strict foreign-key dependency order
TABLES_IN_ORDER = [
    ("sectors", "sectors.csv"),
    ("securities", "securities.csv"),
    ("benchmarks", "benchmarks.csv"),
    ("portfolios", "portfolios.csv"),
    ("holdings", "holdings.csv"),
    ("transactions", "transactions.csv"),
    ("historical_prices", "historical_prices.csv"),
    ("portfolio_performance", "portfolio_performance.csv"),
    ("risk_metrics", "risk_metrics.csv"),
]


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a SQLite database connection with foreign key enforcement enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_database(db_path: str = DEFAULT_DB_PATH, force_reload: bool = False) -> sqlite3.Connection:
    """
    Create the SQLite database using database_schema.sql and load all 9 CSV files.
    
    Supports both persistent file-based SQLite ('portfolio_database.db' as shown in
    README Step 3 & Technical Hints) and in-memory SQLite (':memory:' as mentioned
    in README Step 1).
    """
    is_memory = db_path == ":memory:"

    if not is_memory and force_reload and os.path.exists(db_path):
        os.remove(db_path)

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Execute schema SQL to create tables and indices
    with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
        schema_sql = schema_file.read()
    cursor.executescript(schema_sql)

    # Re-enable foreign keys after executescript (SQLite resets pragma state on script execution if uncommitted)
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Load each CSV into its respective table in dependency order
    for table_name, csv_filename in TABLES_IN_ORDER:
        # Check if table already has data (skip if not reloading)
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        existing_count = cursor.fetchone()[0]
        if existing_count > 0 and not force_reload and not is_memory:
            continue

        if force_reload and not is_memory:
            cursor.execute(f"DELETE FROM {table_name};")

        csv_path = os.path.join(DATA_DIR, csv_filename)
        df = pd.read_csv(csv_path)

        # Convert NaN values to None so SQLite stores NULL properly
        df = df.where(pd.notnull(df), None)

        columns = list(df.columns)
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join(columns)
        insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"

        records = [tuple(None if pd.isna(val) else val for val in row) for row in df.to_numpy()]
        cursor.executemany(insert_sql, records)

    conn.commit()

    # Check foreign key integrity
    cursor.execute("PRAGMA foreign_key_check;")
    fk_errors = cursor.fetchall()
    if fk_errors:
        raise RuntimeError(f"Foreign key constraint violations found: {fk_errors}")

    return conn


def verify_database(conn: sqlite3.Connection) -> None:
    """Verify all 9 tables, row counts, foreign key status, and run sample SELECT queries."""
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys;")
    fk_status = cursor.fetchone()[0]
    print(f"Foreign Keys Enforced (PRAGMA foreign_keys): {bool(fk_status)}")

    print("\n--- Table Row Counts ---")
    print(f"{'Table Name':<25} | {'Row Count':<10}")
    print("-" * 40)
    total_rows = 0
    for table_name, _ in TABLES_IN_ORDER:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        total_rows += count
        print(f"{table_name:<25} | {count:<10}")
    print("-" * 40)
    print(f"{'Total Rows Across 9 Tables':<25} | {total_rows:<10}")

    print("\n--- Sample Query 1: Active Portfolios (First 5) ---")
    cursor.execute("""
        SELECT portfolio_id, portfolio_name, target_risk_level, total_aum, status
        FROM portfolios
        WHERE status = 'Active'
        LIMIT 5;
    """)
    for row in cursor.fetchall():
        print(row)

    print("\n--- Sample Query 2: Technology Sector Securities ---")
    cursor.execute("""
        SELECT s.symbol, s.company_name, s.asset_type, sec.sector_name, s.current_price
        FROM securities s
        JOIN sectors sec ON s.sector_id = sec.sector_id
        WHERE sec.sector_name = 'Technology';
    """)
    for row in cursor.fetchall():
        print(row)

    print("\n--- Sample Query 3: Top 5 Holdings by Cost Basis ---")
    cursor.execute("""
        SELECT p.portfolio_name, s.symbol, h.quantity, h.current_weight, h.cost_basis
        FROM holdings h
        JOIN portfolios p ON h.portfolio_id = p.portfolio_id
        JOIN securities s ON h.security_id = s.security_id
        ORDER BY h.cost_basis DESC
        LIMIT 5;
    """)
    for row in cursor.fetchall():
        print(row)


if __name__ == "__main__":
    print(f"Initializing SQLite database at: {DEFAULT_DB_PATH}")
    connection = initialize_database(DEFAULT_DB_PATH, force_reload=True)
    verify_database(connection)
    connection.close()
    print("\nDatabase setup and verification completed successfully.")
