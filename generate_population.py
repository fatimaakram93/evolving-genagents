import os
import json
import uuid
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from InitialPopulation import InitialPopulation

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'genagents', 'agent_bank', 'populations', 'fifty_agents')

# --- Main Execution ---
def main():
    """Main function to generate the population of agents."""
    logging.info("Starting agent population generation...")
    
    # Instantiate the population generator
    population_generator = InitialPopulation(output_dir=OUTPUT_DIR)
    
    # Generate a population of 1 for testing
    population_generator.create_population(size=50)
            
    logging.info("Agent population generation complete.")

if __name__ == "__main__":
    main()
