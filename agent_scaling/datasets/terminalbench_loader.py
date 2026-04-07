"""Extract Terminal-Bench 2.0 task metadata from the Terminal-Bench repository.

Tasks live on the dataset/terminal-bench-core/v0.1.x branch under ./tasks/.
Each task has a task.yaml with instruction, difficulty, category, etc.,
a Dockerfile for the container image, and test scripts for evaluation.

Usage:
    python -m agent_scaling.datasets.terminalbench_loader --output datasets/terminalbench.json
"""

import argparse
import json
import os
import subprocess
import tempfile

import yaml
from loguru import logger

REPO_URL = "https://github.com/laude-institute/terminal-bench.git"
TASKS_BRANCH = "dataset/terminal-bench-core/v0.1.x"
EXPECTED_TASK_COUNT = 86


def clone_repo(clone_dir: str) -> str:
    """Clone the Terminal-Bench repository (tasks branch)."""
    logger.info(f"Cloning {REPO_URL} branch={TASKS_BRANCH} into {clone_dir}...")
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "--branch", TASKS_BRANCH,
            REPO_URL, clone_dir,
        ],
        check=True,
        capture_output=True,
    )
    return clone_dir


def parse_task_directory(task_dir: str, task_id: str) -> dict:
    """Parse a single task directory into an instance dict."""
    instance = {"task_id": task_id}

    # Read task.yaml (primary config file)
    task_yaml_path = os.path.join(task_dir, "task.yaml")
    if not os.path.exists(task_yaml_path):
        logger.warning(f"No task.yaml found for task {task_id}")
        return instance

    with open(task_yaml_path, "r") as f:
        task_config = yaml.safe_load(f)

    if task_config is None:
        return instance

    instance["instruction"] = task_config.get("instruction", "")
    instance["difficulty"] = task_config.get("difficulty")
    instance["category"] = task_config.get("category")
    instance["time_limit"] = int(task_config.get("max_agent_timeout_sec", 360))

    # Docker image: tasks use ghcr.io images, built from Dockerfile
    dockerfile_path = os.path.join(task_dir, "Dockerfile")
    if os.path.exists(dockerfile_path):
        # Image will be built as terminalbench/{task_id}
        instance["docker_image"] = f"terminalbench/{task_id}"
        with open(dockerfile_path, "r") as f:
            instance["_dockerfile"] = f.read()
    else:
        instance["docker_image"] = "ghcr.io/laude-institute/t-bench/python-3-13:latest"

    # Read test script (run-tests.sh)
    run_tests_path = os.path.join(task_dir, "run-tests.sh")
    if os.path.exists(run_tests_path):
        with open(run_tests_path, "r") as f:
            instance["test_script"] = f.read().strip()

    # Read pytest test file if present
    tests_dir = os.path.join(task_dir, "tests")
    if os.path.isdir(tests_dir):
        for fname in sorted(os.listdir(tests_dir)):
            if fname.startswith("test_") and fname.endswith(".py"):
                fpath = os.path.join(tests_dir, fname)
                with open(fpath, "r") as f:
                    instance["test_script_pytest"] = f.read().strip()
                break

    # Read reference solution if available
    solution_path = os.path.join(task_dir, "solution.sh")
    if os.path.exists(solution_path):
        with open(solution_path, "r") as f:
            instance["reference_solution"] = f.read().strip()

    return instance


def extract_tasks(repo_dir: str) -> list:
    """Extract all tasks from the cloned repository."""
    instances = []
    tasks_dir = os.path.join(repo_dir, "tasks")

    if not os.path.isdir(tasks_dir):
        logger.error(f"Tasks directory not found at {tasks_dir}")
        return instances

    for entry in sorted(os.listdir(tasks_dir)):
        entry_path = os.path.join(tasks_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.startswith("."):
            continue

        instance = parse_task_directory(entry_path, entry)
        if instance.get("instruction"):
            # Remove internal fields before saving
            instance.pop("_dockerfile", None)
            instances.append(instance)

    return instances


def load_and_convert(output_path: str) -> None:
    """Clone repo, extract tasks, and save as framework JSON."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = os.path.join(tmp_dir, "terminal-bench")
        clone_repo(repo_dir)
        instances = extract_tasks(repo_dir)

    logger.info(f"Extracted {len(instances)} tasks (expected: ~{EXPECTED_TASK_COUNT})")

    output = {
        "dataset_id": "terminalbench",
        "instances": instances,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(instances)} instances to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Terminal-Bench tasks and convert to framework JSON format"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    args = parser.parse_args()
    load_and_convert(args.output)


if __name__ == "__main__":
    main()
