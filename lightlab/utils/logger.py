import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(funcName)s:line %(lineno)d: %(message)s",
)
LOGGER = logging.getLogger(__name__)
