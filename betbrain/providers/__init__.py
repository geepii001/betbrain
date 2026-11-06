"""Data providers. Each returns DataFrames in the schema of db.upsert_matches/odds/injuries.

Add a provider by writing a function returning (matches, odds, injuries) DataFrames.
Scraping WhoScored/Flashscore is intentionally not included (their terms prohibit it and
they actively block bots); use a licensed API or CSV feed through one of these adapters.
"""
