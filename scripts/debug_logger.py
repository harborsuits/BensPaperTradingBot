import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/Users/bendickinson/Desktop/Trading:BenBot/api_log_20250504_193724.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set OpenAI debug logging
logging.getLogger("openai").setLevel(logging.DEBUG)
