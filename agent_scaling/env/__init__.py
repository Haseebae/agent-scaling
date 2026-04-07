from .base import AgentEnvironment
from .basic import BasicEnvironment
try:
    from .browsecomp import BrowseCompPlusEnvironment
except ImportError:
    pass  # tevatron/torch not installed — browsecomp env unavailable

# from .browsecomp import BrowseCompPlusEnvironment
from .plancraft import PlancraftEnvironment
from .registry import (
    T,
    get_env,
    get_env_cls,
    is_env_registered,
    list_envs,
    register_env,
)
from .swebench import SWEBenchEnvironment
from .terminalbench import TerminalBenchEnvironment
from .web_search import WebSearchEnvironment
