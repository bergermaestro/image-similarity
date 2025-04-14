import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create a logger instance
logger = logging.getLogger("funding_programs_logger")

# Example usage
if __name__ == "__main__":
    logger.info("Logger is set up and ready to use!")
