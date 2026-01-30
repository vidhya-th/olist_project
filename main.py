"""from src.logger import logger

if __name__ == "__main__":
    logger.info("Starting OlistPricing app")
    """



"""
Olist Dynamic Pricing Pipeline - Fixed main.py
"""
from src.logger import logger  #  Your global logger - already configured
from src.utils import load_master_dataset, extract_flash_sale_features, engineer_pricing_features
from src.exception import OlistException

def run_pipeline():
    """Main pipeline orchestrator"""
    logger.info(" Starting Olist Dynamic Pricing Pipeline")
    
    try:
        # Step 1: Load master dataset (112k x 22)
        logger.info(" Loading master dataset...")
        df = load_master_dataset()
        
        # Step 2: Flash sale timing features
        logger.info(" Extracting flash sale features...")
        df = extract_flash_sale_features(df)
        
        # Step 3: Pricing model features
        logger.info(" Engineering pricing features...")
        df = engineer_pricing_features(df)
        
        logger.info(f" Pipeline completed: {df.shape}")
        return True
        
    except OlistException as e:
        logger.error(f" Business error: {str(e)}")
        return False
        
    except Exception as e:
        logger.error(f" Unexpected error: {str(e)}")
        logger.exception("Full traceback:")
        return False

def main():
    """Entry point"""
    success = run_pipeline()
    if success:
        logger.info(" Pipeline SUCCESS!")
        exit(0)
    else:
        logger.error(" Pipeline FAILED!")
        exit(1)

if __name__ == "__main__":
    main()


