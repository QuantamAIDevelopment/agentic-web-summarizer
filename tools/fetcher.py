import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException, Timeout, HTTPError
import logging
from datetime import datetime

# Set up optional logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def fetch_url_content(url: str, timeout: int = 10) -> str:
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/112.0.0.0 Safari/537.36"
        )
    }

    try:
        logger.info(f"Fetching URL: {url}")
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except Timeout:
        logger.error(f"Timeout occurred when fetching: {url}")
        raise ValueError(f"Timeout occurred when fetching {url}")
    except HTTPError as e:
        logger.error(f"HTTP error while fetching {url}: {e}")
        raise ValueError(f"HTTP error: {e}")
    except RequestException as e:
        logger.error(f"Network error while fetching {url}: {e}")
        raise ValueError(f"Network error: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, and other non-content elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Extract visible text and clean it up
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)
    
    # Log the fetch operation to the database
    try:
        from models.agent_log_utils import log_agent_decision
        log_agent_decision(
            agent_name="fetcher_tool",
            input_data=url,
            output_summary=f"Fetched {len(clean_text)} characters",
            reflection=f"Successfully retrieved content from {url}",
            reasoning_chain=[f"Fetched URL: {url}", f"Retrieved {len(response.text)} raw characters", f"Extracted {len(clean_text)} clean characters"]
        )
    except Exception as e:
        logger.error(f"Failed to log fetch operation: {e}")

    logger.info(f"Fetched {len(clean_text)} characters from {url}")
    return clean_text
