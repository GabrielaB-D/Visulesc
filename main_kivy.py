import sys
import os

# Asegurar import de src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ui.kivy_app import run_kivy


def main():
    run_kivy()


if __name__ == "__main__":
    main()


