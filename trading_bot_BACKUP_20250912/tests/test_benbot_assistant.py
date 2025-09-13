import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock modules before import
sys.modules['trading_bot.data.data_manager'] = MagicMock()
sys.modules['trading_bot.learning.backtest_learner'] = MagicMock()
sys.modules['trading_bot.dashboard.backtest_dashboard'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trading_bot.assistant.benbot_assistant import BenBotAssistant


class TestBenBotAssistant(unittest.TestCase):
    
    def setUp(self):
        # Create mock objects for dependencies
        self.mock_data_manager = MagicMock()
        self.mock_dashboard = MagicMock()
        
        # Initialize the assistant with mock dependencies
        self.assistant = BenBotAssistant(
            data_manager=self.mock_data_manager,
            dashboard_interface=self.mock_dashboard
        )
    
    def test_process_message_help(self):
        # Test help message processing
        response = self.assistant.process_message("help")
        self.assertIn("Available commands", response)
    
    def test_process_message_backtest(self):
        # Setup mock for backtest query
        self.mock_data_manager.get_backtest_info.return_value = {
            "strategy_name": "TestStrategy",
            "performance": {"return": 0.15, "sharpe": 1.2}
        }
        
        # Test backtest query
        response = self.assistant.process_message("show me backtest results for TestStrategy")
        self.assertIn("TestStrategy", response)
    
    def test_process_message_models(self):
        # Setup mock for model query
        self.mock_data_manager.get_model_info.return_value = {
            "model_name": "RandomForest",
            "accuracy": 0.85
        }
        
        # Test model query
        response = self.assistant.process_message("how is the RandomForest model performing?")
        self.assertIn("RandomForest", response)
    
    def test_process_message_dashboard(self):
        # Setup mock for dashboard query
        self.mock_dashboard.get_current_view.return_value = "backtest_results"
        
        # Test dashboard query
        response = self.assistant.process_message("what's on the dashboard?")
        self.assertIn("dashboard", response.lower())
    
    def test_process_message_general(self):
        # Test general conversation
        response = self.assistant.process_message("hello")
        self.assertTrue(response)  # Just ensure we get a response
    

if __name__ == "__main__":
    unittest.main() 