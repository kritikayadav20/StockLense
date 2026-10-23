"""User-facing errors. Messages are written to be shown in the dashboard."""


class StockLensError(Exception):
    """Base error whose text is safe to show to a user."""


class DataValidationError(StockLensError):
    """The request itself is not usable, for example a ticker or period."""


class DataQualityError(StockLensError):
    """Downloaded or supplied rows failed validation."""


class DataAccessError(StockLensError):
    """The market-data provider could not be reached or returned nothing."""


class ModelTrainingError(StockLensError):
    """The model could not be trained or scored on the available history."""


class SentimentModelError(StockLensError):
    """FinBERT could not be loaded or did not return the expected classes."""


class NewsError(StockLensError):
    """The news provider request failed."""


class LLMError(StockLensError):
    """The optional language-model request failed."""


class DatabaseError(StockLensError):
    """SQLite could not be read or written."""
