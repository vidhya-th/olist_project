
from src.logger import logger  # Your global "OlistPricing" logger 

class OlistException(Exception):
    """
    Base custom exception for Olist Dynamic Pricing project.
    Automatically logs full traceback + business context using your global logger.
    
    Usage:
        raise OlistException(original_error, "Data loading failed for file X")
    """
    
    def __init__(self, error: Exception, context: str = ""):
        self.original_error = error
        self.context = context
        
        # Build descriptive message
        if context:
            message = f"[CONTEXT: {context}] {error.__class__.__name__}: {str(error)}"
        else:
            message = f"{error.__class__.__name__}: {str(error)}"
            
        super().__init__(message)
        
        # Log with FULL traceback using YOUR global logger
        logger.error(f" OlistException: {message}")
        logger.exception("   Full traceback:", exc_info=error)

# Specific Olist exceptions (inherit from base)
class DataValidationError(OlistException):
    """Raised for data quality/validation issues"""
    pass

class ModelTrainingError(OlistException):
    """Raised during model training/prediction failures"""
    pass

class PricingPolicyError(OlistException):
    """Raised when pricing policy constraints violated"""
    pass