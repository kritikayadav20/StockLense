# Data directory

StockLens does not ship fabricated historical prices, news, or financial statements.

When you run the app, downloaded Yahoo Finance history is cached in `data/cache/` and a small SQLite file is created at `data/stocklens.db` (or the path in `DATABASE_PATH`). Both are local working files and are listed in `.gitignore`.

If a download fails and no cache exists, the app shows an error and asks you to retry. It does not fill the gap with invented prices.
