"""Containerized solver execution for ChipLab."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chiplab.schemas.common import TrackedValue, Provenance
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.common import FidelityLevel


class DockerRunner:
    """Run solvers in Docker containers."""

    def __init__(self, docker_client: Optional[Any] = None):
        """Initialize Docker runner.

        Args:
            docker_client: Docker SDK client instance (if None, will try to use docker CLI)
        """
        self.docker_client = docker_client

    def pull_image(self, image_name: str, tag: str = "latest") -> bool:
        """Pull a Docker image.

        Args:
            image_name: Name of the Docker image
            tag: Image tag

        Returns:
            True if successfully pulled
        """
        try:
            if self.docker_client:
                self.docker_client.images.pull(image_name, tag=tag)
            else:
                result = subprocess.run(
                    ["docker", "pull", f"{image_name}:{tag}"],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    return False
            return True
        except Exception:
            return False

    def run_container(
        self,
        image: str,
        command: List[str],
        volumes: Optional[List[Tuple[str, str]]] = None,
        gpu: bool = False,
        memory_limit: Optional[str] = None,
        cpu_limit: Optional[str] = None,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess:
        """Run a solver in a Docker container.

        Args:
            image: Docker image name
            command: Command to run inside the container
            volumes: List of (host_path, container_path) volume mounts
            gpu: Whether to enable GPU passthrough
            memory_limit: Memory limit (e.g., "4g")
            cpu_limit: CPU limit (e.g., "2.0")
            timeout: Timeout in seconds

        Returns:
            CompletedProcess from running the container
        """
        cmd = ["docker", "run"]

        # Add GPU support if requested
        if gpu:
            cmd.extend(["--gpus", "all"])

        # Add memory limit
        if memory_limit:
            cmd.extend(["--memory", memory_limit])

        # Add CPU limit
        if cpu_limit:
            cmd.extend(["--cpus", cpu_limit])

        # Add volume mounts
        if volumes:
            for host_path, container_path in volumes:
                cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Add timeout
        cmd.extend(["--timeout", str(timeout)])

        # Add the image and command
        cmd.append(image)
        cmd.extend(command)

        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return process
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Docker container timed out after {timeout}s")
        except FileNotFoundError:
            raise RuntimeError("Docker not available")

    def build_image(
        self,
        dockerfile_path: str,
        tag: str,
        build_args: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Build a Docker image from a Dockerfile.

        Args:
            dockerfile_path: Path to the Dockerfile
            tag: Tag for the built image
            build_args: Build arguments for the Dockerfile

        Returns:
            True if successfully built
        """
        try:
            if self.docker_client:
                build_kwargs = {"path": str(dockerfile_path), "tag": tag}
                if build_args:
                    build_kwargs["buildargs"] = build_args
                self.docker_client.images.build(**build_kwargs)
            else:
                build_cmd = ["docker", "build", "-t", tag, str(dockerfile_path)]
                if build_args:
                    for key, value in build_args.items():
                        build_cmd.extend(["--build-arg", f"{key}={value}"])
                result = subprocess.run(build_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return False
            return True
        except Exception:
            return False


class SingularityRunner:
    """Run solvers in Singularity containers for HPC environments."""

    def __init__(self):
        """Initialize Singularity runner."""
        pass

    def run_container(
        self,
        image_path: str,
        command: List[str],
        bind_paths: Optional[List[str]] = None,
        gpu: bool = False,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess:
        """Run a solver in a Singularity container.

        Args:
            image_path: Path to the Singularity image file
            command: Command to run inside the container
            bind_paths: List of paths to bind mount
            gpu: Whether to enable GPU passthrough
            timeout: Timeout in seconds

        Returns:
            CompletedProcess from running the container
        """
        cmd = ["singularity", "run"]

        if bind_paths:
            for path in bind_paths:
                cmd.extend(["--bind", path])

        if gpu:
            # HPC GPU support - simplified
            cmd.extend(["--nv"])

        cmd.append(image_path)
        cmd.extend(command)

        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return process
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Singularity container timed out after {timeout}s")
        except FileNotFoundError:
            raise RuntimeError("Singularity not available")


class EnvironmentManager:
    """Manage conda/env modules for native solver execution."""

    def __init__(self):
        """Initialize environment manager."""
        pass

    def create_environment(
        self,
        env_name: str,
        packages: List[str],
        python_version: str = "3.11",
    ) -> str:
        """Create a conda environment with required packages.

        Args:
            env_name: Name for the conda environment
            packages: List of conda packages to install
            python_version: Python version string

        Returns:
            Path to the created environment
        """
        import subprocess
        try:
            # Check if environment already exists
            result = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True, text=True
            )
            envs = [line.strip() for line in result.stdout.split("\n") if line.strip()]

            if env_name not in envs:
                subprocess.run(
                    ["conda", "create", "--name", env_name,
                     f"python={python_version}"] + packages,
                    capture_output=True, text=True, check=True
                )
            return f"/home/everett/miniconda3/envs/{env_name}"
        except Exception as e:
            raise RuntimeError(f"Failed to create conda environment: {e}")

    def activate_environment(self, env_path: str) -> Dict[str, str]:
        """Activate a conda environment and return environment variables.

        Args:
            env_path: Path to the conda environment

        Returns:
            Dictionary of environment variables
        """
        return {
            "CONDA_DEFAULT_ENV": env_path.split("/")[-1],
            "PATH": f"{env_path}/bin:{__import__('os').pathsep}/usr/bin:/usr/local/bin",
        }

    def is_package_installed(self, env_name: str, package_name: str) -> bool:
        """Check if a package is installed in a conda environment.

        Args:
            env_name: Name of the conda environment
            package_name: Name of the package to check

        Returns:
            True if the package is installed
        """
        import subprocess
        try:
            result = subprocess.run(
                ["conda", "list", "-n", env_name, package_name],
                capture_output=True, text=True
            )
            return result.returncode == 0 and package_name in result.stdout
        except Exception:
            return False


class SolverImageBuilder:
    """Dockerfiles for each solver with ChipLab integration."""

    @staticmethod
    def get_lammps_dockerfile() -> str:
        """Get Dockerfile for LAMMPS with ChipLab integration.

        Returns:
            Dockerfile content as a string
        """
        return """FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    g++ \\
    make \\
    gcc \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install LAMMPS
RUN wget https://github.com/lammps/lammps/releases/download/lammps_2024_06_27/lammps-20240627.tar.gz \\
    && tar -xzf lammps-20240627.tar.gz \\
    && cd lammps-20240627 \\
    && mkdir build \\
    && cd build \\
    && cmake .. -DLAMMPS_MANUAL_PACKAGE=ON \\
    && make -j$(nproc) \\
    && make install

# Install ChipLab dependencies
RUN pip3 install numpy pandas pyyaml

# Set working directory
WORKDIR /chiplab

# Copy ChipLab solver scripts
COPY chiplab_solver_scripts /chiplab/chiplab_solver_scripts

ENV PATH="/chiplab/chiplab_solver_scripts:${PATH}"

CMD [\"lammps\"]
"""

    @staticmethod
    def get_spice_dockerfile() -> str:
        """Get Dockerfile for SPICE simulators with ChipLab integration.

        Returns:
            Dockerfile content as a string
        """
        return """FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    gcc \\
    make \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install ngspice
RUN wget http://ngspice.sourceforge.net/release/ngspice-35.tar.gz \\
    && tar -xzf ngspice-35.tar.gz \\
    && cd ngspice-35 \\
    && ./configure --with-x \\
    && make \\
    && make install

# Install ChipLab dependencies
RUN pip3 install numpy pandas pyyaml

# Set working directory
WORKDIR /chiplab

# Copy ChipLab solver scripts
COPY chiplab_solver_scripts /chiplab/chiplab_solver_scripts

ENV PATH="/chiplab/chiplab_solver_scripts:${PATH}"

CMD [\"ngspice\"]
"""

    @staticmethod
    def get_openroad_dockerfile() -> str:
        """Get Dockerfile for OpenROAD with ChipLab integration.

        Returns:
            Dockerfile content as a string
        """
        return """FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    git \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install OpenROAD flow
RUN git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-templates.git \\
    && cd OpenROAD-flow-templates \\
    && make docker-image \\
    && make flow

# Install ChipLab dependencies
RUN pip3 install numpy pandas pyyaml

# Set working directory
WORKDIR /chiplab

# Copy ChipLab solver scripts
COPY chiplab_solver_scripts /chiplab/chiplab_solver_scripts

ENV PATH="/chiplab/chiplab_solver_scripts:${PATH}"

CMD [\"openroad\"]
"""
