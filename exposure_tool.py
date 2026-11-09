"""Exposure Calculator Tool for calculating portfolio sector exposures (equities only)."""

import sqlite3
from typing import Dict, Any


def resolve_portfolio_name(db_connection: sqlite3.Connection, portfolio_name: str) -> Dict[str, Any]:
    """
    Resolve a user-supplied portfolio name against the portfolios table using:
    1. Exact case-insensitive matching
    2. Unique partial case-insensitive matching
    Returns {'portfolio_id': int, 'portfolio_name': str} on success, or a structured
    error dictionary if empty, ambiguous, or not found.
    """
    if not portfolio_name or not isinstance(portfolio_name, str) or not portfolio_name.strip():
        return {
            "error": "Portfolio name cannot be empty.",
            "user_message": "Please provide a valid portfolio name.",
        }

    if db_connection is None:
        return {
            "error": "Database connection is not available.",
            "user_message": "Unable to resolve portfolio because the database is currently unavailable.",
        }

    cleaned_name = portfolio_name.strip()

    try:
        cursor = db_connection.cursor()

        # 1. Exact case-insensitive match takes priority
        cursor.execute(
            "SELECT portfolio_id, portfolio_name FROM portfolios WHERE LOWER(portfolio_name) = LOWER(?);",
            (cleaned_name,),
        )
        exact_row = cursor.fetchone()

        if exact_row:
            return {
                "portfolio_id": exact_row[0],
                "portfolio_name": exact_row[1],
            }

        # 2. Partial case-insensitive match
        cursor.execute(
            "SELECT portfolio_id, portfolio_name FROM portfolios "
            "WHERE LOWER(portfolio_name) LIKE LOWER(?) ORDER BY portfolio_name ASC;",
            (f"%{cleaned_name}%",),
        )
        partial_rows = cursor.fetchall()

        if len(partial_rows) == 1:
            return {
                "portfolio_id": partial_rows[0][0],
                "portfolio_name": partial_rows[0][1],
            }
        elif len(partial_rows) > 1:
            matched_names = [row[1] for row in partial_rows]
            names_str = ", ".join(matched_names)
            return {
                "error": f"Ambiguous portfolio name '{cleaned_name}'. Matches: {names_str}",
                "matching_portfolios": matched_names,
                "user_message": (
                    f"Multiple portfolios match '{cleaned_name}': {names_str}. "
                    "Please clarify by specifying the exact portfolio name."
                ),
            }
        else:
            return {
                "error": f"Portfolio '{cleaned_name}' not found in the database.",
                "user_message": f"Portfolio '{cleaned_name}' was not found. Please check the portfolio name and try again.",
            }
    except sqlite3.Error as exc:
        return {
            "error": f"Database error during portfolio resolution: {str(exc)}",
            "user_message": "A database error occurred while resolving the portfolio name.",
        }
    except Exception as exc:
        return {
            "error": f"Unexpected error during portfolio resolution: {str(exc)}",
            "user_message": "An unexpected error occurred while resolving the portfolio name.",
        }


def calculate_sector_exposure(db_connection: sqlite3.Connection, portfolio_name: str) -> Dict[str, Any]:
    """
    Calculate sector percentage exposures across equity ('Stock') holdings only,
    ignoring bond holdings.
    """
    if not portfolio_name or not isinstance(portfolio_name, str) or not portfolio_name.strip():
        return {
            "error": "Portfolio name cannot be empty.",
            "user_message": "Please provide a valid portfolio name to calculate sector exposure.",
        }

    if db_connection is None:
        return {
            "error": "Database connection is not available.",
            "user_message": "Unable to calculate sector exposure because the database is currently unavailable.",
        }

    resolution = resolve_portfolio_name(db_connection, portfolio_name)
    if "error" in resolution:
        return resolution

    portfolio_id = resolution["portfolio_id"]
    resolved_name = resolution["portfolio_name"]

    try:
        cursor = db_connection.cursor()
        query = """
            SELECT sec.sector_name,
                   ROUND(SUM(h.current_weight) * 100, 2) AS exposure_percentage
            FROM holdings h
            JOIN securities s ON h.security_id = s.security_id
            JOIN sectors sec ON s.sector_id = sec.sector_id
            WHERE h.portfolio_id = ?
              AND s.asset_type = 'Stock'
            GROUP BY sec.sector_name
            ORDER BY exposure_percentage DESC;
        """
        cursor.execute(query, (portfolio_id,))
        results = cursor.fetchall()

        exposures = {sector: pct for sector, pct in results}
        return {
            "portfolio_id": portfolio_id,
            "portfolio_name": resolved_name,
            "sector_exposures": exposures,
        }

    except sqlite3.Error as exc:
        return {
            "error": f"Database error during sector exposure calculation: {str(exc)}",
            "user_message": "A database error occurred while calculating sector exposures.",
        }
    except Exception as exc:
        return {
            "error": f"Unexpected error during sector exposure calculation: {str(exc)}",
            "user_message": "An unexpected error occurred while calculating sector exposures.",
        }
