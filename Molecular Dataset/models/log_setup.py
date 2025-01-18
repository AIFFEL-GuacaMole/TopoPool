import logging

def init_logger():
    # Create a logger instance
    logger = logging.getLogger("debugLogger")
    logger.setLevel(logging.DEBUG)

    debug_handler = logging.FileHandler("debug.log")
    debug_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(formatter)
    
    logger.addHandler(debug_handler)
    logger.propagate = False
    
    return logger

logger = init_logger()
