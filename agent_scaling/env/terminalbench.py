"""Terminal-Bench 2.0 Docker environment with bash and submit tools."""

import threading
from typing import Dict, Optional

from loguru import logger

from agent_scaling.datasets.terminalbench import TerminalBenchInstance
from agent_scaling.env.base import AgentEnvironmentTools
from agent_scaling.env.docker_utils import DockerContainer
from agent_scaling.env.registry import register_env
from agent_scaling.env.tools import cls_tool


@register_env("terminalbench")
class TerminalBenchEnvironment(AgentEnvironmentTools):
    """Docker-based environment for Terminal-Bench 2.0 evaluation.

    Provides 2 tools: bash, submit.
    Uses network_mode="bridge" since tasks may require network access.
    Multiple agent environments can share the same Docker container for
    multi-agent setups via class-level container caching.
    """

    # Class-level container cache for multi-agent sharing
    _container_cache: Dict[str, DockerContainer] = {}
    _container_locks: Dict[str, threading.Lock] = {}
    _cache_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_instance: Optional[TerminalBenchInstance] = self.dataset_instance
        self.is_done = False
        self.success = False
        self.num_steps = 0
        self._container: Optional[DockerContainer] = None
        self._is_container_owner = False

        if self.dataset_instance is not None:
            self._init_container()

    def _init_container(self) -> None:
        """Initialize or reuse a Docker container for this instance."""
        instance = self.dataset_instance
        assert instance is not None
        task_id = instance.task_id

        with TerminalBenchEnvironment._cache_lock:
            if task_id not in TerminalBenchEnvironment._container_cache:
                container = DockerContainer(
                    image=instance.docker_image,
                    workspace_dir="/",
                    timeout=instance.time_limit,
                    memory_limit="8g",
                    network_mode="bridge",  # Terminal-Bench tasks may need network
                )
                container.start()

                # Auto-detect workspace: use /app if it exists, otherwise /
                exit_code, _, _ = container.exec_command(
                    "test -d /app", timeout=5
                )
                if exit_code == 0:
                    container.workspace_dir = "/app"
                    logger.info(f"Using /app as workspace for task {task_id}")

                TerminalBenchEnvironment._container_cache[task_id] = container
                TerminalBenchEnvironment._container_locks[task_id] = threading.Lock()
                self._is_container_owner = True
                logger.info(f"Created container for task {task_id}")
            else:
                logger.info(f"Reusing container for task {task_id}")

            self._container = TerminalBenchEnvironment._container_cache[task_id]

    def get_instance_prompt_info(self) -> Dict[str, str]:
        info = {}
        if self.dataset_instance:
            info.update(self.dataset_instance.get_prompt_info())

            # Add workspace context by listing key directories
            if self._container:
                _, stdout, _ = self._container.exec_command(
                    "ls -la / 2>/dev/null | head -30 && echo '---' && ls -la /app/ 2>/dev/null | head -20"
                )
                info["workspace_info"] = stdout

        info["tools_description"] = self.tools_description
        return info

    def env_done(self) -> bool:
        return self.is_done

    def _cleanup(self) -> None:
        """Clean up Docker container if this env is the owner."""
        if not self._is_container_owner or self.dataset_instance is None:
            return

        task_id = self.dataset_instance.task_id
        with TerminalBenchEnvironment._cache_lock:
            container = TerminalBenchEnvironment._container_cache.pop(task_id, None)
            TerminalBenchEnvironment._container_locks.pop(task_id, None)

        if container is not None:
            container.stop_and_remove()
            logger.info(f"Cleaned up container for task {task_id}")

    @cls_tool
    def bash(self, command: str) -> str:
        """Execute a bash command in the container.

        Args:
            command: The bash command to execute.

        Returns:
            Command output (stdout and stderr combined).
        """
        assert self._container is not None, "Container not initialized"
        self.num_steps += 1

        exit_code, stdout, stderr = self._container.exec_command(
            command, timeout=300
        )

        output = stdout
        if stderr:
            output += f"\n[STDERR]\n{stderr}"
        if exit_code != 0:
            output += f"\n[Exit code: {exit_code}]"

        # Truncate long output to prevent context window pollution
        max_output = 8000
        if len(output) > max_output:
            output = (
                output[:2000]
                + f"\n\n... [{len(output) - 7000} characters truncated] ...\n\n"
                + output[-5000:]
            )

        return output

    @cls_tool
    def submit(self, reasoning: str) -> str:
        """Submit the task for evaluation.

        Installs pytest in the container, copies the test file, and runs it
        to verify whether the task was completed correctly.

        Args:
            reasoning: Brief explanation of what was done to complete the task.

        Returns:
            Evaluation result showing pass or fail.
        """
        assert self._container is not None, "Container not initialized"
        assert self.dataset_instance is not None

        self.num_steps += 1
        self.is_done = True

        instance = self.dataset_instance

        # Prefer the actual pytest test file; fall back to wrapper script
        if instance.test_script_pytest:
            return self._run_pytest_eval(instance, reasoning)
        elif instance.test_script:
            return self._run_shell_eval(instance, reasoning)
        else:
            self.success = False
            return (
                "No verification script available for this task. "
                "Result cannot be automatically verified."
            )

    def _run_pytest_eval(self, instance, reasoning: str) -> str:
        """Run evaluation using the pytest test file directly.

        Replicates Harbor's test infrastructure:
        1. Install curl + uv (astral.sh package manager)
        2. Initialize uv project + add pytest
        3. Copy test file and run with uv run pytest
        """
        assert self._container is not None

        # Step 1: Setup test directory
        test_dir = "/tmp/tb_tests"
        self._container.exec_command(f"mkdir -p {test_dir}")

        # Step 2: Write the pytest test file
        self._container.write_file(f"{test_dir}/test_outputs.py", instance.test_script_pytest)

        # Step 3: Install pytest — try multiple methods with verification
        setup_cmd = (
            "export PATH=\"$HOME/.local/bin:/usr/local/bin:$PATH\"; "
            # Try uv first (fastest)
            "if command -v uv >/dev/null 2>&1; then "
            f"  cd {test_dir} && uv init -q 2>/dev/null; uv add pytest -q 2>/dev/null && echo 'UV_OK'; "
            # Try pip/pip3
            "elif command -v pip >/dev/null 2>&1; then "
            "  pip install -q pytest 2>/dev/null && echo 'PIP_OK'; "
            "elif command -v pip3 >/dev/null 2>&1; then "
            "  pip3 install -q pytest 2>/dev/null && echo 'PIP3_OK'; "
            # Try python -m pip
            "elif python3 -m pip --version >/dev/null 2>&1; then "
            "  python3 -m pip install -q pytest 2>/dev/null && echo 'PYMPIP_OK'; "
            # Last resort: install pip via apt or ensurepip, then install pytest
            "else "
            "  (apt-get update -qq 2>/dev/null && apt-get install -y -qq python3-pip 2>/dev/null) || "
            "  (python3 -m ensurepip --default-pip 2>/dev/null) || "
            "  (curl -sS https://bootstrap.pypa.io/get-pip.py | python3 2>/dev/null) || "
            "  (wget -qO- https://bootstrap.pypa.io/get-pip.py | python3 2>/dev/null); "
            "  pip3 install -q pytest 2>/dev/null || pip install -q pytest 2>/dev/null || "
            "  python3 -m pip install -q pytest 2>/dev/null && echo 'BOOTSTRAP_OK'; "
            "fi"
        )
        exit_code, stdout, stderr = self._container.exec_command(
            setup_cmd, timeout=180
        )
        use_uv = "UV_OK" in stdout
        # Verify pytest is actually importable
        verify_exit, verify_out, _ = self._container.exec_command(
            "python3 -c 'import pytest; print(pytest.__version__)' 2>&1 || "
            "python -c 'import pytest; print(pytest.__version__)' 2>&1"
        )
        use_minimal_runner = False
        if verify_exit != 0:
            logger.warning(f"pytest installation failed, falling back to minimal runner: {verify_out}")
            # Fallback: create a minimal test runner that doesn't require pytest
            minimal_runner = '''
import sys, traceback, importlib.util

def run_tests(test_file):
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    mod = importlib.util.module_from_spec(spec)
    passed = 0
    failed = 0
    errors = []
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"FAILED to load test module: {e}")
        sys.exit(1)

    for name in dir(mod):
        if name.startswith("test_"):
            func = getattr(mod, name)
            if callable(func):
                try:
                    func()
                    passed += 1
                    print(f"PASSED: {name}")
                except AssertionError as e:
                    failed += 1
                    errors.append((name, str(e)))
                    print(f"FAILED: {name} - {e}")
                except Exception as e:
                    failed += 1
                    errors.append((name, str(e)))
                    print(f"ERROR: {name} - {e}")

    print(f"\\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    run_tests(sys.argv[1])
'''
            self._container.write_file(f"{test_dir}/minimal_runner.py", minimal_runner)
            use_minimal_runner = True
        else:
            logger.info(f"pytest {verify_out.strip()} available for evaluation")

        # Step 4: Run the pytest test file
        if use_minimal_runner:
            run_cmd = (
                f"cd {test_dir} && python3 {test_dir}/minimal_runner.py {test_dir}/test_outputs.py 2>&1 || "
                f"python {test_dir}/minimal_runner.py {test_dir}/test_outputs.py 2>&1"
            )
        elif use_uv:
            run_cmd = (
                "export PATH=\"$HOME/.local/bin:$PATH\"; "
                f"cd {test_dir} && uv run pytest {test_dir}/test_outputs.py -rA 2>&1"
            )
        else:
            run_cmd = (
                f"cd {test_dir} && python3 -m pytest {test_dir}/test_outputs.py -rA 2>&1 || "
                f"python -m pytest {test_dir}/test_outputs.py -rA 2>&1"
            )
        exit_code, stdout, stderr = self._container.exec_command(
            run_cmd, timeout=300
        )

        self.success = exit_code == 0

        status = "PASSED" if self.success else "FAILED"
        result_text = (
            f"## Evaluation Result: {status}\n\n"
            f"**Reasoning:** {reasoning}\n\n"
            f"**Test output:**\n{stdout}\n"
        )
        if stderr:
            result_text += f"\n**Stderr:**\n{stderr}\n"

        return result_text

    def _run_shell_eval(self, instance, reasoning: str) -> str:
        """Fallback: run the shell test script directly."""
        assert self._container is not None

        self._container.write_file("/tmp/test_script.sh", instance.test_script)

        exit_code, stdout, stderr = self._container.exec_command(
            "chmod +x /tmp/test_script.sh && /tmp/test_script.sh 2>&1",
            timeout=300,
        )

        self.success = exit_code == 0

        status = "PASSED" if self.success else "FAILED"
        result_text = (
            f"## Evaluation Result: {status}\n\n"
            f"**Reasoning:** {reasoning}\n\n"
            f"**Test script output:**\n{stdout}\n"
        )
        if stderr:
            result_text += f"\n**Stderr:**\n{stderr}\n"

        return result_text
