import sys
import uuid
import axelrod as axl
sys.path.append('/Users/fatima.akram/Documents/genagents')
from genagents import GenerativeAgent
import openai
import os
import time
import inspect
import logging
import instructor
import json
from pydantic import BaseModel
from typing import Optional

# ========== Setup Logging ==========
openai.debug = True
log_dir = "/Users/fatima.akram/Documents/logs"
os.makedirs(log_dir, exist_ok=True)
timestr = time.strftime("%Y%m%d-%H%M%S")

logging.basicConfig(
    filename=os.path.join(log_dir, f"match_log_{timestr}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ========== API Key ==========
client = instructor.patch(openai.OpenAI(api_key="api-key"))

# ========== Evolved Agent ==========
EVOLVED_BASE_DIR = "./evolved_agents"
os.makedirs(EVOLVED_BASE_DIR, exist_ok=True)

# ========== Agent ==========
AGENT_PATH = "/Users/fatima.akram/Documents/genagents/agent_bank/populations/single_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae"

agent = GenerativeAgent(agent_folder=AGENT_PATH)

class GenAgentPlayer(axl.Player):
    name = "GenAgentPlayer"

    def __init__(self):
        super().__init__()
        self.genagent = agent
        self.time_step = 1

    def strategy(self, opponent):
        
        current_turn = len(self.history) + 1
        logger.info(f"Turn {current_turn}: Opponent has {'no last move yet.' if len(opponent.history) == 0 else f'last move: {opponent.history[-1]}'}")
        # Record opponent's last move if any
        if len(opponent.history) > 0:
            last_move = "cooperated" if opponent.history[-1] == axl.Action.C else "defected"
            self.genagent.remember(f"Opponent {last_move} in the last round", time_step=self.time_step)
        
        self.time_step += 1
        
        # Every 2 rounds, tweak personality trait
        if self.time_step % 1 == 0:
            self.suggest_trait_change()

        # Make decision
        # In GA, you must make changes in the fitness function after the whole match/tournament, and not in between rounds
        questions = {
            "Should I cooperate with my opponent?": ["Yes", "No"]
        }
        try:
            response = self.genagent.categorical_resp(questions)
            logger.info(f"Turn {current_turn}: Response received: {response}")

            action = axl.Action.C  
            
            # Check if response has the expected structure
            if isinstance(response, dict) and "responses" in response:
                responses_list = response["responses"]

                if isinstance(responses_list, list) and len(responses_list) > 0:
                    decision = responses_list[0]
                    logger.info(f"Turn {current_turn}: Decision: {decision}")

                if isinstance(decision, str):
                    action = axl.Action.C if decision.strip().lower() == "yes" else axl.Action.D
                    logger.info(f"Turn {current_turn}: Action chosen: {action}")
            
            logger.info(f"Turn {current_turn}: Final action being returned is {action}.")
            return action 

        except Exception as e:
            logger.error(f"Turn {current_turn}: Error in strategy: {e}")
            return axl.Action.C  # Default to cooperation on error
        

    def suggest_trait_change(self):
        class TraitAdjustment(BaseModel):
            node_id: int
            node_type: str = "trait_adjustment"
            content: str
            importance: int = 85
            created: int
            last_retrieved: int
            pointer_id: None
        
        # Path to nodes.json
        json_path = "/Users/fatima.akram/Documents/genagents/agent_bank/populations/single_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae/memory_stream/nodes.json"
        print("Here is trait change suggestion")

        #These traits are chosen based on the paper Strategies for the Iterated Prisoner’s Dilemma Anagh Malik 2021
        response = client.chat.completions.create(
            model="gpt-4o",
            response_model=TraitAdjustment,
            messages=[
                {"role": "system", "content": "You are a strategy coach for game-theoretic agents."},
                {"role": "user", "content": (
                    "Analyze the agent's IPD strategy and suggest a trait adjustment to maximize points. The point is to win at any cost "
                    "Focus on one of: Nice (cooperation), Retaliatory (punishment), "
                    "Forgiving (reconciliation), Clear (consistency). "
                    "Format: 'Trait: adjustment direction with strategic reasoning'"
                )}
            ]
        )
            
        logger.info(f"Structured trait change: {response}")
        print(f"Structured trait change: {response.model_dump()}")
            
         # Load existing nodes and determine next node ID
        try:
            with open(json_path, 'r') as f:
                nodes = json.load(f)
                print(f"Reading json at {json_path}")
                if not isinstance(nodes, list):
                    nodes = [nodes]
                    
            # Find max node ID
            max_id = max(node.get('node_id', 0) for node in nodes) if nodes else 0
            new_id = max_id + 1
            
            # Set new node ID
            response.node_id = new_id
            
            # Append new node
            nodes.append(response.model_dump())
            
            # Save back to file
            with open(json_path, 'w') as f:
                json.dump(nodes, f, indent=2)
                
            logger.info(f"Added new trait adjustment node with ID: {new_id}")
            print(f"Added new trait adjustment node with ID: {new_id}")
            
        except Exception as e:
            logger.error(f"Error updating nodes.json: {e}")
            print(f"Error updating nodes.json: {e}")

# ========== Match ==========
def run_match():
    logger.info("Setting up match")

    players = [
        axl.TitForTat(),
        GenAgentPlayer()
    ]
    logger.info(f"Players in the match: {[player.name for player in players]}")
    
    # Setup match
    match = axl.Match(
        players=players,
        turns=2
    )
    
    # Play match
    logger.info("Starting match...")
    match.play()
    logger.info("match completed.")
    logger.info(f"match scores: {match.scores()}")
    winner = match.winner()
    logger.info(f"match winner is: {winner}")
    print(f"winner strategy is {winner}")

try:
    print("Testing API connection...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
    logger.info(f"API response: {response}")
    print("API connection successful!")
    
    print("\nInitialising match...")
    run_match()
    
except Exception as e:
    print(f"Error while playing match: {str(e)}")