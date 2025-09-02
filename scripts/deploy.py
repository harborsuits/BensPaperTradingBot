#!/usr/bin/env python3
# scripts/deploy.py

import os
import sys
import argparse
import subprocess
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger("deploy")

# Environment configurations
ENVIRONMENTS = {
    "development": {
        "kube_context": "k8s-dev-cluster",
        "namespace": "trading-bot-dev",
        "config_path": "kubernetes/dev/",
        "requires_approval": False,
    },
    "staging": {
        "kube_context": "k8s-staging-cluster",
        "namespace": "trading-bot-staging",
        "config_path": "kubernetes/staging/",
        "requires_approval": False,
    },
    "production": {
        "kube_context": "k8s-prod-cluster",
        "namespace": "trading-bot-prod",
        "config_path": "kubernetes/prod/",
        "requires_approval": True,
    }
}

def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, text=True, capture_output=True)
    return result

def ensure_kubectl_installed() -> bool:
    """Check if kubectl is installed."""
    try:
        run_command(["kubectl", "version", "--client"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("kubectl is not installed or not in PATH.")
        return False

def ensure_kube_context(context: str) -> bool:
    """Ensure the correct Kubernetes context is set."""
    try:
        current_context = run_command(
            ["kubectl", "config", "current-context"]
        ).stdout.strip()
        
        if current_context != context:
            logger.info(f"Switching context from {current_context} to {context}")
            run_command(["kubectl", "config", "use-context", context])
            
        logger.info(f"Using Kubernetes context: {context}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set Kubernetes context: {e}")
        return False

def deploy_to_environment(env_name: str, component: str = None, tag: str = None) -> bool:
    """Deploy to the specified environment."""
    if env_name not in ENVIRONMENTS:
        logger.error(f"Unknown environment: {env_name}")
        return False
    
    env_config = ENVIRONMENTS[env_name]
    
    # Check if the environment requires manual approval
    if env_config["requires_approval"]:
        approval = input(f"Do you want to deploy to {env_name}? (yes/no): ")
        if approval.lower() != "yes":
            logger.info("Deployment cancelled.")
            return False

    # Ensure kubectl is installed
    if not ensure_kubectl_installed():
        return False
    
    # Ensure the correct Kubernetes context
    if not ensure_kube_context(env_config["kube_context"]):
        return False
    
    # Apply Kubernetes configurations
    config_path = env_config["config_path"]
    logger.info(f"Applying Kubernetes configurations from {config_path}")
    
    try:
        # Apply all configs
        run_command(["kubectl", "apply", "-f", config_path])
        
        # If a specific component is specified, only restart that deployment
        if component:
            deployment = f"trading-bot-{component}"
            logger.info(f"Restarting deployment {deployment}")
            
            # If a tag is provided, update the image first
            if tag:
                image = f"ghcr.io/your-org/trading-bot-{component}:{tag}"
                logger.info(f"Updating image to {image}")
                run_command([
                    "kubectl", "set", "image", 
                    f"deployment/{deployment}", 
                    f"{component}={image}"
                ])
            
            # Restart the deployment
            run_command(["kubectl", "rollout", "restart", f"deployment/{deployment}"])
            
            # Wait for the rollout to complete
            run_command([
                "kubectl", "rollout", "status", 
                f"deployment/{deployment}", 
                "--timeout=300s"
            ])
        else:
            # Restart all deployments
            components = ["api-server", "data-collector", "trading-system", "monitoring-service"]
            for comp in components:
                deployment = f"trading-bot-{comp}"
                logger.info(f"Restarting deployment {deployment}")
                run_command(["kubectl", "rollout", "restart", f"deployment/{deployment}"])
                run_command([
                    "kubectl", "rollout", "status", 
                    f"deployment/{deployment}", 
                    "--timeout=300s"
                ])
        
        logger.info(f"Deployment to {env_name} completed successfully.")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy trading bot to a specific environment.")
    
    parser.add_argument(
        "--environment", "-e",
        choices=list(ENVIRONMENTS.keys()),
        required=True,
        help="The environment to deploy to"
    )
    
    parser.add_argument(
        "--component", "-c",
        choices=["api-server", "data-collector", "trading-system", "monitoring-service"],
        help="Specific component to deploy (default: all components)"
    )
    
    parser.add_argument(
        "--tag", "-t",
        help="Specific image tag to deploy (default: use the tag from Kubernetes manifest)"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    success = deploy_to_environment(
        env_name=args.environment,
        component=args.component,
        tag=args.tag
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 