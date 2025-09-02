#!/usr/bin/env python3
"""
Update EvoTrader with Smart Features

This script updates the main forex_evotrader.py file to incorporate all smart modules:
- Adds import statements for smart modules
- Updates the ForexEvoTrader class initialization to create smart components
- Enhances the CLI to expose smart functionality
- Performs a backup of the original file before making changes
"""

import os
import sys
import shutil
import re
import argparse
from datetime import datetime

# File paths
EVOTRADER_PATH = '/Users/bendickinson/Desktop/Evotrader/forex_evotrader.py'
BACKUP_FOLDER = '/Users/bendickinson/Desktop/Evotrader/backups'


def create_backup():
    """Create a backup of the original forex_evotrader.py file"""
    if not os.path.exists(BACKUP_FOLDER):
        os.makedirs(BACKUP_FOLDER)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(BACKUP_FOLDER, f'forex_evotrader_{timestamp}.py')
    
    shutil.copy2(EVOTRADER_PATH, backup_path)
    print(f"Created backup at: {backup_path}")
    return backup_path


def update_imports(content):
    """Add import statements for smart modules if they don't exist"""
    smart_imports = [
        "from forex_smart_session import SmartSessionAnalyzer",
        "from forex_smart_pips import SmartPipAnalyzer",
        "from forex_smart_news import SmartNewsAnalyzer",
        "from forex_smart_compliance import SmartComplianceMonitor",
        "from forex_smart_benbot import SmartBenBotConnector",
        "from forex_smart_integration import ForexSmartIntegration"
    ]
    
    # Check if imports already exist
    for import_line in smart_imports:
        if import_line not in content:
            # Find the last import statement
            import_matches = list(re.finditer(r'^import|^from', content, re.MULTILINE))
            if import_matches:
                last_import_pos = import_matches[-1].start()
                last_import_line_end = content.find('\n', last_import_pos) + 1
                
                # Insert our imports after the last import
                content = (
                    content[:last_import_line_end] + 
                    '\n' + '\n'.join(smart_imports) + '\n' +
                    content[last_import_line_end:]
                )
                break
    
    return content


def update_init_method(content):
    """Update the __init__ method to add smart component initialization"""
    smart_init_code = """
        # Initialize smart components
        self.smart_session = SmartSessionAnalyzer(self.session_performance_tracker, self.config)
        self.smart_pips = SmartPipAnalyzer(self.pair_manager, self.config)
        self.smart_news = SmartNewsAnalyzer(self.news_guard, self.config)
        self.smart_compliance = SmartComplianceMonitor(self.config)
        self.smart_benbot = SmartBenBotConnector(
            self.config.get('benbot_endpoint', 'http://localhost:8080/benbot/api'), 
            self.config)
        
        # Create smart integration
        self.smart_integration = ForexSmartIntegration(self, self.config)
        
        # Flag to track if methods have been enhanced
        self._smart_methods_enhanced = False
        
        logger.info("Smart forex components initialized")
    """
    
    # Find the end of the __init__ method
    init_pattern = r'def __init__\(self,.*?\):'
    init_match = re.search(init_pattern, content, re.DOTALL)
    
    if init_match:
        # Find the end of the method by balancing indentation
        init_start = init_match.end()
        init_body_start = content.find('\n', init_start) + 1
        
        # Find the base indentation for the method body
        init_line = content[init_match.start():init_body_start].strip()
        next_line = content[init_body_start:content.find('\n', init_body_start)].rstrip()
        base_indent = len(next_line) - len(next_line.lstrip())
        
        # Find the end of the method by looking for the next line with the same or less indentation
        lines = content[init_body_start:].split('\n')
        method_end_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                method_end_idx = i
                break
        
        if method_end_idx == 0:
            # Couldn't find end of method, so put at end of file
            method_end_pos = len(content)
        else:
            # Found the end of the method
            method_end_pos = init_body_start + sum(len(lines[j]) + 1 for j in range(method_end_idx))
        
        # Only add if not already added
        if "smart_integration" not in content[:method_end_pos]:
            # Format the smart init code with the correct indentation
            indent_str = ' ' * base_indent
            smart_code_indented = '\n'.join(
                indent_str + line if line.strip() else line 
                for line in smart_init_code.split('\n')
            )
            
            # Insert our code before the end of the method
            content = content[:method_end_pos] + smart_code_indented + content[method_end_pos:]
    
    return content


def add_enhance_methods_method(content):
    """Add a method to enhance the standard methods with smart functionality"""
    enhance_method = """
    def enhance_with_smart_methods(self):
        \"\"\"
        Enhance standard methods with smart functionality.
        This should be called after initializing the object.
        \"\"\"
        if self._smart_methods_enhanced:
            logger.info("Smart methods already enhanced")
            return
        
        # Store original methods
        self._original_methods = {
            'check_session_optimal': self.check_session_optimal,
            'calculate_pip_target': self.calculate_pip_target,
            'check_news_safe': self.check_news_safe,
            'calculate_position_size': self.calculate_position_size,
            'consult_benbot': self.consult_benbot
        }
        
        # Replace with enhanced methods from smart integration
        self.check_session_optimal = self.smart_integration.enhanced_check_session_optimal
        self.calculate_pip_target = self.smart_integration.enhanced_calculate_pip_target
        self.check_news_safe = self.smart_integration.enhanced_check_news_safe
        self.calculate_position_size = self.smart_integration.enhanced_calculate_position_size
        self.consult_benbot = self.smart_integration.enhanced_consult_benbot
        
        self._smart_methods_enhanced = True
        logger.info("Enhanced standard methods with smart functionality")
    
    def revert_to_standard_methods(self):
        \"\"\"Revert to original standard methods.\"\"\"
        if not self._smart_methods_enhanced:
            logger.info("Smart methods were not enhanced")
            return
        
        # Restore original methods
        self.check_session_optimal = self._original_methods['check_session_optimal']
        self.calculate_pip_target = self._original_methods['calculate_pip_target']
        self.check_news_safe = self._original_methods['check_news_safe']
        self.calculate_position_size = self._original_methods['calculate_position_size']
        self.consult_benbot = self._original_methods['consult_benbot']
        
        self._smart_methods_enhanced = False
        logger.info("Reverted to standard methods")
    """
    
    # Find the class definition
    class_pattern = r'class ForexEvoTrader\('
    class_match = re.search(class_pattern, content)
    
    if class_match:
        # Find a good place to add the new method
        # Ideally after all the existing methods but before any CLI code
        
        # Look for the CLI section or the end of the class
        cli_pattern = r'def main\(\):|if __name__ == "__main__":'
        cli_match = re.search(cli_pattern, content)
        
        if cli_match:
            # Found CLI section, add before it
            insert_pos = content.rfind('\n\n', 0, cli_match.start())
            if insert_pos == -1:
                insert_pos = cli_match.start()
        else:
            # Add at the end of the file
            insert_pos = len(content)
        
        # Only add if not already added
        if "enhance_with_smart_methods" not in content:
            # Insert our methods
            content = content[:insert_pos] + enhance_method + content[insert_pos:]
    
    return content


def update_cli(content):
    """Update the CLI to add smart functionality"""
    smart_cli_code = """
    # Smart analysis parser
    smart_parser = subparsers.add_parser(
        "smart-analysis", 
        help="Run smart analysis on a pair or strategy")
    
    smart_parser.add_argument(
        "--pair", 
        type=str, 
        required=True, 
        help="Currency pair to analyze")
        
    smart_parser.add_argument(
        "--strategy-id", 
        type=str, 
        help="Strategy ID to analyze")
        
    smart_parser.add_argument(
        "--analysis-type", 
        type=str,
        choices=["session", "pips", "news", "compliance", "all"],
        default="all",
        help="Type of smart analysis to run")
    """
    
    # Find the CLI argument parser section
    parser_pattern = r'subparsers = parser\.add_subparsers\('
    parser_match = re.search(parser_pattern, content)
    
    if parser_match:
        # Find a good place to add the new parser
        # Look for the last parser.add_argument call
        last_parser_pos = content.rfind('add_parser', parser_match.start())
        if last_parser_pos != -1:
            # Find the end of this parser's argument definitions
            next_section_pos = content.find('\n\n', last_parser_pos)
            if next_section_pos == -1:
                next_section_pos = len(content)
            
            # Only add if not already added
            if "smart-analysis" not in content[:next_section_pos]:
                # Insert our CLI code
                content = content[:next_section_pos] + smart_cli_code + content[next_section_pos:]
    
    return content


def add_cli_handler(content):
    """Add CLI handler for smart analysis command"""
    smart_handler = """
    elif args.command == "smart-analysis":
        evo_trader = ForexEvoTrader(args.config)
        
        # Ensure smart methods are enhanced
        evo_trader.enhance_with_smart_methods()
        
        # Run smart analysis
        results = evo_trader.smart_integration.run_smart_analysis(
            args.pair, args.analysis_type, args.strategy_id)
        
        # Print results
        import json
        print(json.dumps(results, indent=2, default=str))
    """
    
    # Find the CLI command handlers
    handler_pattern = r'if args\.command =='
    handler_matches = list(re.finditer(handler_pattern, content))
    
    if handler_matches:
        # Find the last handler
        last_handler = handler_matches[-1]
        last_handler_block_end = content.find('elif', last_handler.start())
        if last_handler_block_end == -1:
            # Look for else block
            last_handler_block_end = content.find('else', last_handler.start())
        
        if last_handler_block_end == -1:
            # Look for next empty line
            last_handler_block_end = content.find('\n\n', last_handler.start())
        
        if last_handler_block_end == -1:
            # Use end of file
            last_handler_block_end = len(content)
        
        # Only add if not already added
        if "smart-analysis" not in content[:last_handler_block_end]:
            # Insert our handler
            content = content[:last_handler_block_end] + smart_handler + content[last_handler_block_end:]
    
    return content


def update_evotrader():
    """Update the forex_evotrader.py file with smart enhancements"""
    # Create backup
    backup_path = create_backup()
    
    # Read current content
    with open(EVOTRADER_PATH, 'r') as f:
        content = f.read()
    
    # Apply updates
    content = update_imports(content)
    content = update_init_method(content)
    content = add_enhance_methods_method(content)
    content = update_cli(content)
    content = add_cli_handler(content)
    
    # Write updated content
    with open(EVOTRADER_PATH, 'w') as f:
        f.write(content)
    
    print(f"Updated {EVOTRADER_PATH} with smart enhancements")
    print(f"Original file backed up to {backup_path}")


def main():
    parser = argparse.ArgumentParser(description='Update EvoTrader with Smart Features')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup (not recommended)')
    
    args = parser.parse_args()
    
    if args.no_backup:
        # Skip backup
        global create_backup
        create_backup = lambda: None
    
    update_evotrader()
    print("Update complete! EvoTrader now has smart capabilities.")
    print("\nTo use smart features:")
    print("1. Initialize normally: evo_trader = ForexEvoTrader()")
    print("2. Enable smart methods: evo_trader.enhance_with_smart_methods()")
    print("3. Use the enhanced methods as you would the standard ones")
    print("\nOr via CLI: python forex_evotrader.py smart-analysis --pair EURUSD")


if __name__ == "__main__":
    main()
