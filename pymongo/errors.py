"""
Mock pymongo errors module for testing without MongoDB dependencies
"""

class PyMongoError(Exception):
    """Base class for all PyMongo exceptions."""
    pass

class ConnectionFailure(PyMongoError):
    """Raised when a connection to the database cannot be made or is lost."""
    pass

class OperationFailure(PyMongoError):
    """Raised when a database operation fails."""
    pass

class DuplicateKeyError(OperationFailure):
    """Raised when an insert or update fails due to a duplicate key error."""
    pass

class BulkWriteError(OperationFailure):
    """Raised when a bulk write operation fails."""
    pass

class InvalidOperation(PyMongoError):
    """Raised when a client attempts to perform an invalid operation."""
    pass

class ConfigurationError(PyMongoError):
    """Raised when something is incorrectly configured."""
    pass

class ServerSelectionTimeoutError(ConnectionFailure):
    """Raised when the driver cannot find an available server."""
    pass

class NetworkTimeout(PyMongoError):
    """Raised when a network operation times out."""
    pass
