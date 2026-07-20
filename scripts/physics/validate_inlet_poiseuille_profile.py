"""CLI wrapper for raw FEM inlet Poiseuille-profile diagnostics."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.physics.cfd.inlet_profile_diagnostic import main


if __name__ == "__main__":
    main()
