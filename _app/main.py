#!/usr/bin/env python3
"""
Enhanced Football Betting System - Main CLI Entry Point
Phase 3: Clean, maintainable command interface

Usage:
    python main.py <command> [options]

Commands:
    predict     Generate enhanced predictions with PPI and betting analysis
    analyze     Analyze team/league performance using domain services
    status      Show system status and data availability
    help        Show this help message

Examples:
    python main.py predict --league Premier-League --verbose
    python main.py analyze team Arsenal --league Premier-League --season 2023-2024
    python main.py status
"""

import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

# Add the app directory to Python path for imports
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))


def print_main_help():
    """Print main help message"""
    help_text = """
🚀 Enhanced Football Betting System - Phase 3 Architecture

USAGE:
    python main.py <command> [options]

COMMANDS:
    predict     Generate enhanced predictions with PPI analysis and betting edges
                Uses domain services to create comprehensive match analysis
                
    analyze     Analyze team/league performance using domain services  
                Provides detailed PPI calculations and performance metrics
                
    status      Show system status, data availability, and service health
                Quick diagnostic tool for troubleshooting
                
    help        Show detailed help for commands

EXAMPLES:
    # Generate predictions for all leagues
    python main.py predict
    
    # Generate predictions for specific league with higher edge threshold
    python main.py predict --league Premier-League --min-edge 0.03
    
    # Analyze specific team performance
    python main.py analyze team Arsenal --league Premier-League --season 2023-2024
    
    # Show league PPI rankings
    python main.py analyze rankings Premier-League --season 2023-2024
    
    # Check system status
    python main.py status

HELP:
    python main.py <command> --help     # Detailed help for specific command
    python main.py help                 # This help message

PHASE 3 FEATURES:
    ✅ Clean domain services (PPI Calculator, Edge Calculator)
    ✅ Enhanced prediction workflows  
    ✅ Proper error handling and user feedback
    ✅ Structured output with confidence scores
    ✅ CSV export capabilities
    ✅ Betting opportunity identification
    ✅ Performance analysis tools

For more information, see the documentation or run specific command help.
    """
    print(help_text)


def main():
    """Main entry point for the enhanced betting system"""

    if len(sys.argv) < 2:
        print("❌ No command specified.")
        print_main_help()
        return 1

    command = sys.argv[1].lower()
    args = sys.argv[2:] if len(sys.argv) > 2 else []

    # Handle help commands
    if command in ["help", "--help", "-h"]:
        print_main_help()
        return 0

    # Route to appropriate command handler
    try:
        if command == "predict":
            from interfaces.cli.commands.prediction_commands import PredictCommand

            handler = PredictCommand()
            return handler.handle(args)

        elif command == "analyze":
            from interfaces.cli.commands.data_commands import AnalyzeCommand

            handler = AnalyzeCommand()
            return handler.handle(args)

        elif command == "status":
            from interfaces.cli.commands.status_command import StatusCommand

            handler = StatusCommand()
            return handler.handle(args)

        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: predict, analyze, status, help")
            return 1

    except ImportError as e:
        print(f"❌ Failed to import command handler: {e}")
        print("💡 Make sure all Phase 3 files are in place and Python path is correct")
        return 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 130

    except Exception as e:
        print(f"❌ Command failed: {e}")
        if "--verbose" in args or "-v" in args:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
