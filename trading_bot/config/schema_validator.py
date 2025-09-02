#!/usr/bin/env python3
"""
Configuration Schema Validator

Validates configuration files against JSON Schema definitions and provides
detailed error reporting. This module is used both by the CI/CD pipeline
and by the runtime configuration loading to ensure configuration validity.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import jsonschema
from jsonschema import validators, ValidationError
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("config_validator")

# Root directory
ROOT_DIR = Path(__file__).parent.parent.parent


class SchemaValidationError(Exception):
    """Exception raised for schema validation errors."""
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class SchemaValidationResult(BaseModel):
    """Result of schema validation."""
    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    file_path: str
    schema_path: str


def extend_with_default(validator_class):
    """Extend the jsonschema validator to set default values."""
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema and not isinstance(instance, list):
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )


DefaultValidatingValidator = extend_with_default(jsonschema.Draft7Validator)


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON in {file_path}: {str(e)}")
    except IOError as e:
        raise SchemaValidationError(f"Error opening {file_path}: {str(e)}")


def validate_config_against_schema(
    config_path: Path, 
    schema_path: Path
) -> SchemaValidationResult:
    """
    Validate a config file against a JSON schema.
    
    Args:
        config_path: Path to the configuration file
        schema_path: Path to the JSON schema file
        
    Returns:
        ValidationResult object with validation status and any errors
    """
    result = SchemaValidationResult(
        is_valid=True,
        file_path=str(config_path),
        schema_path=str(schema_path)
    )
    
    try:
        # Load schema and config
        schema = load_json_file(schema_path)
        
        if config_path.suffix.lower() in ('.json', '.jsonc'):
            config = load_json_file(config_path)
        elif config_path.suffix.lower() in ('.yaml', '.yml'):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                raise SchemaValidationError(f"Error parsing YAML file {config_path}: {str(e)}")
        else:
            raise SchemaValidationError(f"Unsupported file format: {config_path.suffix}")
        
        # Create validator that will fill in defaults
        validator = DefaultValidatingValidator(schema)
        
        # Collect errors
        errors = list(validator.iter_errors(config))
        
        if errors:
            result.is_valid = False
            for error in errors:
                error_path = " -> ".join([str(p) for p in error.path]) if error.path else "root"
                result.errors.append(f"{error_path}: {error.message}")
                
        # Check for additional warnings (like unknown fields)
        unknown_props = find_unknown_properties(config, schema)
        for path, prop in unknown_props:
            path_str = ".".join(path) if path else "root"
            result.warnings.append(f"Unknown property '{prop}' in {path_str}")
                
    except SchemaValidationError as e:
        result.is_valid = False
        result.errors.append(str(e))
        if e.errors:
            result.errors.extend(e.errors)
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Unexpected error during validation: {str(e)}")
        
    return result


def find_unknown_properties(
    config: Dict[str, Any], 
    schema: Dict[str, Any], 
    path: List[str] = None
) -> List[Tuple[List[str], str]]:
    """Find properties in the config that aren't in the schema."""
    path = path or []
    unknown_props = []
    
    # Check if this is an object with properties
    if "properties" in schema and isinstance(config, dict):
        properties = schema.get("properties", {})
        
        # Check for extra properties
        for prop in config:
            if prop not in properties:
                unknown_props.append((path.copy(), prop))
                
        # Recurse into nested properties
        for prop, value in config.items():
            if prop in properties and isinstance(value, (dict, list)):
                if isinstance(value, dict):
                    unknown_props.extend(
                        find_unknown_properties(
                            value, 
                            properties[prop], 
                            path + [prop]
                        )
                    )
                elif isinstance(value, list) and "items" in properties[prop]:
                    # For arrays, check each item
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            unknown_props.extend(
                                find_unknown_properties(
                                    item, 
                                    properties[prop]["items"], 
                                    path + [prop, str(i)]
                                )
                            )
    
    return unknown_props


def validate_all_configs(base_dir: Path = None) -> Dict[str, SchemaValidationResult]:
    """
    Validate all configuration files against their schemas.
    
    Args:
        base_dir: Base directory to search for config files, defaults to config directory
        
    Returns:
        Dictionary mapping file paths to validation results
    """
    base_dir = base_dir or ROOT_DIR / "config"
    results = {}
    
    # Primary configuration schema
    schema_path = base_dir / "config.schema.json"
    
    if not schema_path.exists():
        logger.warning(f"Schema file not found: {schema_path}")
        return results
    
    # Find all config files
    for config_path in base_dir.glob("**/*.yaml"):
        if config_path.name.startswith("_") or config_path.name.startswith("."):
            continue
        
        logger.info(f"Validating {config_path}")
        result = validate_config_against_schema(config_path, schema_path)
        results[str(config_path)] = result
        
        if not result.is_valid:
            logger.error(f"Validation failed for {config_path}:")
            for error in result.errors:
                logger.error(f"  - {error}")
        
        if result.warnings:
            logger.warning(f"Warnings for {config_path}:")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
    
    # Also check JSON configs
    for config_path in base_dir.glob("**/*.json"):
        # Skip schema files
        if config_path.name.endswith(".schema.json") or config_path.name.startswith("_"):
            continue
        
        logger.info(f"Validating {config_path}")
        result = validate_config_against_schema(config_path, schema_path)
        results[str(config_path)] = result
        
        if not result.is_valid:
            logger.error(f"Validation failed for {config_path}:")
            for error in result.errors:
                logger.error(f"  - {error}")
        
        if result.warnings:
            logger.warning(f"Warnings for {config_path}:")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
    
    return results


def main():
    """Main entry point for the validation script."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="BensBot Configuration Validator")
    parser.add_argument("--config", type=str, help="Path to configuration file to validate")
    parser.add_argument("--schema", type=str, help="Path to schema file")
    parser.add_argument("--base-dir", type=str, help="Base directory for config files")
    args = parser.parse_args()
    
    exit_code = 0
    
    try:
        if args.config:
            # Validate a specific config file
            config_path = Path(args.config)
            schema_path = Path(args.schema) if args.schema else ROOT_DIR / "config" / "config.schema.json"
            
            logger.info(f"Validating {config_path} against {schema_path}")
            result = validate_config_against_schema(config_path, schema_path)
            
            if not result.is_valid:
                logger.error(f"Validation failed for {config_path}:")
                for error in result.errors:
                    logger.error(f"  - {error}")
                exit_code = 1
            else:
                logger.info(f"✅ {config_path} is valid")
                
            if result.warnings:
                logger.warning(f"Warnings for {config_path}:")
                for warning in result.warnings:
                    logger.warning(f"  - {warning}")
        else:
            # Validate all config files
            base_dir = Path(args.base_dir) if args.base_dir else ROOT_DIR / "config"
            logger.info(f"Validating all configuration files in {base_dir}")
            
            results = validate_all_configs(base_dir)
            
            # Check if any validations failed
            invalid_configs = [
                path for path, result in results.items() 
                if not result.is_valid
            ]
            
            if invalid_configs:
                logger.error(f"❌ {len(invalid_configs)} configuration files failed validation:")
                for path in invalid_configs:
                    logger.error(f"  - {path}")
                exit_code = 1
            else:
                logger.info(f"✅ All {len(results)} configuration files are valid")
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
