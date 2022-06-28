from aulepredictor import aule
import argparse
import sys

def parse_cli():
    p = argparse.ArgumentParser()
    from pathlib import Path # Provisional code to access pre-trained models.

    path = Path(__file__).parent / "../data/test.csv"
    with path.open() as f:
        test = list(csv.reader(f))


if __name__ == '__main__':
    message = "Using AulePredictor by Raúl Fernández-Díaz"
    print("-" * (len(message) + 4))
    print("| " + message + " |")
    print("-" * (len(message) + 4))

    
    Aule = aule()

