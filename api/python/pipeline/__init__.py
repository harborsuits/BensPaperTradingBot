#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Package for Trading Bot

This package provides end-to-end pipeline automation for the trading bot,
including data ingestion, feature engineering, model training, evaluation,
and deployment.
"""

from .pipeline_runner import PipelineRunner

__all__ = ['PipelineRunner'] 