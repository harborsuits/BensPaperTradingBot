#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Package for Trading Bot

This package provides REST API endpoints for model prediction and deployment.
"""

# Expose key components
from .prediction_service import app

__all__ = ['app'] 