#!/usr/bin/env python3
"""
Unit tests for the dashboard launcher script.

These tests validate the functionality of the dashboard launcher,
including argument parsing and command execution.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, call, ANY
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import module to test
from trading_bot.run_dashboard import main

class TestDashboardLauncher(unittest.TestCase):
    """Tests for the dashboard launcher script"""
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('subprocess.Popen')
    @patch('os.environ')
    def test_main_with_default_args(self, mock_environ, mock_popen, mock_exists, mock_parse_args):
        """Test running main with default arguments"""
        # Setup mock arguments
        mock_args = argparse.Namespace(
            port=8501,
            alpaca_key=None,
            alpaca_secret=None,
            mock_data=False
        )
        mock_parse_args.return_value = mock_args
        
        # Make sure the dashboard file exists
        mock_exists.return_value = True
        
        # Mock the subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Call the function
        main()
        
        # Verify the subprocess was called correctly
        mock_popen.assert_called_once()
        
        # Extract the command that was called
        cmd = mock_popen.call_args[0][0]
        
        # Verify it contains the expected elements
        self.assertIn('streamlit', cmd)
        self.assertIn('run', cmd)
        self.assertIn('live_trading_dashboard.py', cmd[-1])
        self.assertIn('--server.port', cmd)
        self.assertIn('8501', cmd)
        
        # Verify the process wait was called
        mock_process.wait.assert_called_once()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('subprocess.Popen')
    @patch('os.environ')
    def test_main_with_custom_port(self, mock_environ, mock_popen, mock_exists, mock_parse_args):
        """Test running main with custom port"""
        # Setup mock arguments with custom port
        mock_args = argparse.Namespace(
            port=9000,
            alpaca_key=None,
            alpaca_secret=None,
            mock_data=False
        )
        mock_parse_args.return_value = mock_args
        
        # Make sure the dashboard file exists
        mock_exists.return_value = True
        
        # Mock the subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Call the function
        main()
        
        # Extract the command that was called
        cmd = mock_popen.call_args[0][0]
        
        # Verify it contains the custom port
        self.assertIn('--server.port', cmd)
        self.assertIn('9000', cmd)
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('subprocess.Popen')
    @patch('os.environ')
    def test_main_with_alpaca_credentials(self, mock_environ, mock_popen, mock_exists, mock_parse_args):
        """Test running main with Alpaca credentials"""
        # Setup mock arguments with Alpaca credentials
        mock_args = argparse.Namespace(
            port=8501,
            alpaca_key="test_key",
            alpaca_secret="test_secret",
            mock_data=False
        )
        mock_parse_args.return_value = mock_args
        
        # Make sure the dashboard file exists
        mock_exists.return_value = True
        
        # Mock the subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Call the function
        main()
        
        # Verify environment variables were set
        mock_environ.__setitem__.assert_any_call("ALPACA_API_KEY", "test_key")
        mock_environ.__setitem__.assert_any_call("ALPACA_API_SECRET", "test_secret")
        
        # Verify the subprocess was called
        mock_popen.assert_called_once()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('logging.error')
    @patch('sys.exit')
    def test_main_with_missing_dashboard_file(self, mock_exit, mock_log_error, mock_exists, mock_parse_args):
        """Test running main when dashboard file doesn't exist"""
        # Setup mock arguments
        mock_args = argparse.Namespace(
            port=8501,
            alpaca_key=None,
            alpaca_secret=None,
            mock_data=False
        )
        mock_parse_args.return_value = mock_args
        
        # Make sure the dashboard file doesn't exist
        mock_exists.return_value = False
        
        # Call the function
        main()
        
        # Verify error was logged and exit was called
        mock_log_error.assert_called_once()
        mock_exit.assert_called_once_with(1)
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('subprocess.Popen')
    @patch('logging.info')
    def test_main_with_mock_data(self, mock_log_info, mock_popen, mock_exists, mock_parse_args):
        """Test running main with mock data flag"""
        # Setup mock arguments with mock data flag
        mock_args = argparse.Namespace(
            port=8501,
            alpaca_key=None,
            alpaca_secret=None,
            mock_data=True
        )
        mock_parse_args.return_value = mock_args
        
        # Make sure the dashboard file exists
        mock_exists.return_value = True
        
        # Mock the subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Call the function
        main()
        
        # Verify mock data message was logged
        mock_log_info.assert_any_call("Using mock data for demonstration")
        
        # Verify the subprocess was called
        mock_popen.assert_called_once()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('subprocess.Popen')
    @patch('logging.error')
    @patch('sys.exit')
    def test_main_with_subprocess_error(self, mock_exit, mock_log_error, mock_popen, mock_exists, mock_parse_args):
        """Test running main when subprocess raises an error"""
        # Setup mock arguments
        mock_args = argparse.Namespace(
            port=8501,
            alpaca_key=None,
            alpaca_secret=None,
            mock_data=False
        )
        mock_parse_args.return_value = mock_args
        
        # Make sure the dashboard file exists
        mock_exists.return_value = True
        
        # Make the subprocess raise an error
        mock_popen.side_effect = Exception("Test error")
        
        # Call the function
        main()
        
        # Verify error was logged and exit was called
        mock_log_error.assert_called_once()
        self.assertIn("Error running dashboard", mock_log_error.call_args[0][0])
        mock_exit.assert_called_once_with(1)
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    @patch('subprocess.Popen')
    @patch('logging.info')
    def test_main_with_keyboard_interrupt(self, mock_log_info, mock_popen, mock_exists, mock_parse_args):
        """Test running main when interrupted by KeyboardInterrupt"""
        # Setup mock arguments
        mock_args = argparse.Namespace(
            port=8501,
            alpaca_key=None,
            alpaca_secret=None,
            mock_data=False
        )
        mock_parse_args.return_value = mock_args
        
        # Make sure the dashboard file exists
        mock_exists.return_value = True
        
        # Make the subprocess raise a KeyboardInterrupt
        mock_popen.side_effect = KeyboardInterrupt()
        
        # Call the function
        main()
        
        # Verify the interrupt was logged
        mock_log_info.assert_any_call("Dashboard stopped by user")

if __name__ == '__main__':
    unittest.main() 