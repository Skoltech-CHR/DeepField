"""Utils for Dataset."""

from ..field.configs import default_config

STATES_KEYWORD = 'States'
WELLS_KEYWORD = 'Wells'


def get_config():
    """Get default config for model loading."""
    return default_config
