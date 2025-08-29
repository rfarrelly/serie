# interfaces/cli/commands/status_command.py
"""
System status command
"""

from pathlib import Path
from typing import List


class StatusCommand:
    """Show system status and configuration"""

    def __init__(self):
        self.name = "status"
        self.description = "Show system status and available data"

    def handle(self, args: List[str]) -> int:
        """Handle status command"""

        print("🚀 Enhanced Football Betting System - Status")
        print("=" * 50)

        # Check data availability
        print("📁 Data Availability:")

        data_files = [
            ("Historical Matches", "historical_ppi_and_odds.csv"),
            ("Current Fixtures", "fixtures_ppi_and_odds.csv"),
            ("Enhanced Predictions", "latest_zsd_enhanced.csv"),
            ("Betting Candidates", "zsd_betting_candidates.csv"),
            ("Team Dictionary", "team_name_dictionary.csv"),
        ]

        for name, filename in data_files:
            if Path(filename).exists():
                try:
                    import pandas as pd

                    df = pd.read_csv(filename, nrows=1)
                    print(f"   ✅ {name}: Available ({len(df.columns)} columns)")
                except Exception:
                    print(f"   ⚠️  {name}: File exists but may be corrupted")
            else:
                print(f"   ❌ {name}: Not found ({filename})")

        print()

        # Test domain services
        print("🔧 Domain Services:")

        try:
            from domain.services.ppi_calculator import PPICalculator

            ppi_calc = PPICalculator()
            print("   ✅ PPI Calculator: Ready")
        except Exception as e:
            print(f"   ❌ PPI Calculator: Error - {e}")

        try:
            from domain.services.edge_calculator import EdgeCalculator

            edge_calc = EdgeCalculator()
            print("   ✅ Edge Calculator: Ready")
        except Exception as e:
            print(f"   ❌ Edge Calculator: Error - {e}")

        try:
            from infrastructure.data.repositories.csv_match_repository import (
                CSVMatchRepository,
            )

            match_repo = CSVMatchRepository()
            print("   ✅ Match Repository: Ready")
        except Exception as e:
            print(f"   ❌ Match Repository: Error - {e}")

        print()

        # Available commands
        print("💻 Available Commands:")
        print("   predict         - Generate enhanced predictions")
        print("   analyze         - Analyze team/league performance")
        print("   status          - Show this status information")
        print("   --help          - Show detailed help")

        return 0
