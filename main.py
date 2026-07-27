"""
Project FORESIGHT
Demand & Inventory Intelligence

Author : Utkarsh Chaudhari
"""

import os
from datetime import datetime


class ProjectInfo:
    PROJECT_NAME = "Project FORESIGHT"
    VERSION = "1.0"

    def __init__(self):
        self.project_path = os.getcwd()
        self.start_time = datetime.now()

    def display(self):
        print("=" * 60)
        print(f"{self.PROJECT_NAME:^60}")
        print("=" * 60)
        print("Demand & Inventory Intelligence")
        print("-" * 60)
        print(f"Version      : {self.VERSION}")
        print(f"Project Path : {self.project_path}")
        print(f"Started At   : {self.start_time.strftime('%d-%m-%Y %H:%M:%S')}")
        print("=" * 60)


def check_project_structure():
    folders = [
        "data",
        "data/raw",
        "data/processed",
        "src",
        "models",
        "reports",
        "outputs",
        "notebooks",
        "app"
    ]

    print("\nChecking Project Structure...\n")

    for folder in folders:
        if os.path.exists(folder):
            print(f"[OK] {folder}")
        else:
            print(f"[Missing] {folder}")


def main():
    project = ProjectInfo()
    project.display()
    check_project_structure()

    print("\nProject setup completed successfully.")
    print("Ready for Data Pipeline Development.")


if __name__ == "__main__":
    main()