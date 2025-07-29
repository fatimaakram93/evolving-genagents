import os
import json
import uuid
import instructor
import openai
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    client = instructor.patch(openai.OpenAI(api_key="api-key"))
except Exception as e:
    logging.error("OpenAI client could not be initialized. Make sure your OPENAI_API_KEY is set.")
    logging.error(e)
    exit()

# --- Pydantic Models for Data Structure ---
class Node(BaseModel):
    node_id: int
    node_type: str = "observation"
    content: str
    importance: int
    created: int
    last_retrieved: int
    pointer_id: Optional[int] = None

class AgentMemory(BaseModel):
    """
    Represents the complete memory of an agent, consisting of a list of nodes.
    Each node contains a piece of dialogue from an interview.
    """
    nodes: List[Node] = Field(..., description="A list of dialogue nodes that make up the agent's memory.")

class InitialPopulation:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.questions = self.get_interview_questions()

    def get_interview_questions(self) -> List[str]:
        return [
            "To start, I would like to begin with a big question: tell me the story of your life. Start from the beginning -- from your childhood, to education, to family and relationships, and to any major life events you may have had.",
            "Some people tell us that they've reached a crossroads at some points in their life where multiple paths were available, and their choice then made a significant difference in defining who they are. What about you? Was there a moment like that for you, and if so, could you tell me the whole story about that from start to finish?",
            "Some people tell us they made a conscious choice or decision in moments like these, while others say it 'just happened'. What about you?",
            "Do you think another person or an organization could have lent a helping hand during moments like this?",
            "Moving to present time, tell me more about family who are important to you. Do you have a partner, or children?",
            "Are there anyone else in your immediate family whom you have not mentioned? Who are they, and what is your relationship with them like?",
            "Tell me about anyone else in your life we haven’t discussed (like friends or romantic partners). Are there people outside of your family who are important to you?",
            "Now let’s talk about your current neighborhood. Tell me all about the neighborhood and area in which you are living now.",
            "Some people say they feel really safe in their neighborhoods, others not so much. How about for you?",
            "Living any place has its ups and downs. Tell me about what it’s been like for you living here.",
            "Tell me about the people who live with you right now, even people who are staying here temporarily.",
            "Is anyone in your household a temporary member, or is everyone a permanent member of the household?",
            "Tell me about anyone else who stays here from time to time, if there is anyone like that",
            "Is there anyone who usually lives here but is away – traveling, at school, or in a hospital?",
            "Right now, across a typical week, how do your days vary?",
            "At what kind of job or jobs do you work, and when do you work?",
            "Do you have other routines or responsibilities that you did not already share?",
            "Tell me about any recent changes to your daily routine.",
            "If you have children, tell me about a typical weekday during the school year for your children. What is their daily routine? And what are after school activities your children participate in?",
            "Some people we’ve talked to tell us about experiences with law enforcement. How about for you?",
            "What other experiences with law enforcement stand out in your mind?",
            "Some people tell us about experiences of arrest – of loved ones, family members, friends, or themselves. How about for you?",
            "Some people say they vote in every election, some tell us they don’t vote at all. How about you?",
            "How would you describe your political views?",
            "Tell me about any recent changes in your political views.",
            "One topic a lot of people have been talking about recently is race and/or racism and policing. Some tell us issues raised by the Black Lives Matter movement have affected them a lot, some say they’ve affected them somewhat, others say they haven’t affected them at all. How about for you?",
            "How have you been thinking about the issues Black Lives Matter raises?",
            "How have you been thinking about race in the U.S. recently?",
            "How have you responded to the increased focus on race and/or racism and policing? Some people tell us they’re aware of the issue but keep their thoughts to themselves, others say they’ve talked to family and friends, others have joined protests. What about you?",
            "What about people you’re close to? How have they responded to the increased focus on race and/or racism and policing?",
            "Now we’d like to learn more about your health. First, tell me all about your health.",
            "For you, what makes it easy or hard to stay healthy?",
            "Tell me about anything big that has happened in the past two years related to your health: any medical diagnoses, flare-ups of chronic conditions, broken bones, pain – anything like that.",
            "Sometimes, health problems get in the way. They can even affect people’s ability to work or care for their children. How about you?",
            "Sometimes, it’s not your health problem, but the health of a loved one. Has this been an issue for you?",
            "Tell me what it has been like trying to get the health care you or your immediate family need. Have you ever had to forgo getting the health care you need?",
            "Have you or your immediate family ever used alternative forms of medicine? This might include indigenous, non-western, or informal forms of care.",
            "During tough times, some people tell us they cope by smoking or drinking. How about for you?",
            "Other people say they cope by relying on prescriptions, pain medications, marijuana, or other substances. How about for you and can you describe your most recent experience of using them, if any?",
            "How are you and your family coping with, or paying for, your health care needs right now?",
            "Some people are excited about medical vaccination, and others, not so much. How about you?",
            "What are your trusted sources of information about the vaccine?",
            "Now we’re going to talk a bit more about what life was like for you over the past year. Tell me all about how you have been feeling.",
            "Tell me a story about a time in the last year when you were in a rough place or struggling emotionally.",
            "Some people say they struggle with depression, anxiety, or something else like that. How about for you?",
            "How has it been for your family?",
            "Some people say that religion or spirituality is important in their lives, for others not so much. How about you?",
            "Some people tell us they use Facebook, Instagram, or other social media to stay connected with the wider world. How about for you? Tell me all about how you use these types of platforms.",
            "Some people say they use Facebook or Twitter to get (emotional or financial) support during tough times. What about you?",
            "Tell me about any recent changes in your level of stress, worry, and your emotional coping strategies.",
            "This might be repetitive, but before we go on to the next section, I want to quickly make sure I have this right: who are you living with right now?",
            "Any babies or small children?",
            "Anyone who usually lives here but is away – traveling, at school, or in a hospital?",
            "Any lodgers, boarders, or employees who live here?",
            "Who shares the responsibility for the rent, mortgage, or other household expenses?",
            "Now we’d like to talk about how you make ends meet and what the monthly budget looks like for you and your family. What were your biggest expenses last month?",
            "How much did your household spend in the past month? Is that the usual amount? And if not, how much do you usually spend and on what?",
            "Does your household own or rent your home?",
            "Bills can fluctuate over the course of a year. Seasons change, there are holidays and special events, school starts and ends, and so on. Tell me all about how your bills have fluctuated over the past year.",
            "Tell me all about how you coped with any extra expenses in the past months.",
            "Some people have a savings account, some people save in a different way, and some people say they don’t save. How about you?",
            "Some people save for big things, like a home, while others save for a rainy day. How about for you (in the last year)?",
            "Some people have student loans or credit card debt. Others take out loans from family or friends or find other ways of borrowing money. Tell me about all the debts you’re paying on right now.",
            "Tell me about any time during the past year that you haven’t had enough money to buy something that you needed or pay a bill that was due.",
            "What is or was your occupation? In this occupation, what kind of work do you do and what are the most important activities or duties?",
            "Was your occupation covered by a union or employee association contract?",
            "In the past year, how many weeks did you work (for a few hours, including paid vacation, paid sick leave, and military service)?",
            "How many hours each week did you usually work?",
            "In the past week, did you work for pay (even for 1 hour) at a job or business?",
            "Did you have more than one job or business, including part time, evening, or weekend work?",
            "Now, in the past month, how many jobs did you work, including part time, evening, or weekend work?",
            "For your job or occupations, how much money did you receive in the past month?",
            "How often do you get paid?",
            "In total, how much did your household make in the past month?",
            "Switching gears a bit, let’s talk a little about tax time. Did you file taxes last year (the most recent year)?",
            "Tell me more about your workplace. How long have you been at your current job? How would you describe the benefits that come with your job? (This could include things like health benefits, paid time off, vacation, and sick time.)",
            "How would you describe your relationships at work? (How is your relationship with your manager or boss? How are your relationships with your coworkers?)",
            "Tell me about how predictable your work schedule is. How would you describe your job in terms of flexibility with your hours?",
            "Some people say it’s hard to take or keep a job because of child care. How about for you?",
            "Do you receive any payments or benefits from SNAP, food stamps, housing voucher payments, supplemental security income, or any other programs like that?",
            "Tell me about times over the last year when your income was especially low. And tell me about all the things you did to make ends meet during that time.",
            "What would it be like for you if you had to spend $400 for an emergency? Would you have the money, and if not, how would you get it?",
            "Overall, how do you feel about your financial situation?",
            "Are you now married, widowed, divorced, separated, or have you never been married? If you are not currently married, are you currently living with a romantic partner?",
            "For our records, we also need your date of birth. Could you please provide that?",
            "Were you born in the United States? If you were not born in the U.S., what country were you born, and what year did you first come to the U.S. to live?",
            "What city and state were you born in?",
            "Are you of Hispanic, Latinx, or Spanish origin?",
            "What race or races do you identify with?",
            "What is the highest degree or grade you’ve completed?",
            "Are you enrolled in school?",
            "Have you been enrolled in school during the past 3 months?",
            "Have you ever served on active duty in the U.S. Armed Forces, Reserves, or National Guard?",
            "What religion do you identify with, if any?",
            "Generally speaking, do you usually think of yourself as a Democrat, a Republican, an Independent, or what? And how strongly do you associate with that party?",
            "Do you think of yourself as closer to the Democratic Party or to the Republican Party?",
            "What was the city and state that you lived in when you were 16 years old?",
            "Did you live with both your own mother and father when you were 16?",
            "Who else did you live with?",
            "What’s the highest degree or grade your dad completed?",
            "What’s the highest degree or grade your mom completed?",
            "Did your mom work for pay for at least a year while you were growing up?",
            "What was her job? What kind of work did she do? What were her most important activities or duties?",
            "What kind of place did she work for? What kind of business was it? What did they make or do where she worked?",
            "Did your dad work for pay for at least a year while you were growing up?",
            "What was his job? What kind of work did he do? What were his most important activities or duties?",
            "What kind of place did he work for? What kind of business was it? What did they make or do where he worked?",
            "We all have hopes about what our future will look like. Imagine yourself a few years from now. Maybe you want your life to be the same in some ways as it is now. Maybe you want it to be different in some ways. What do you hope for?",
            "What do you value the most in your life?",
            "And that was the last question I wanted to ask today. Thank you so much again for your time <participant’s name>. It was really wonderful getting to know you through this interview. Now, we will take you back to the Home Screen so that you may finish the rest of the study!"
        ]

    def generate_agent_memory(self) -> Optional[AgentMemory]:
        """Generates a new agent's memory using the LLM."""
        if not self.questions:
            logging.error("No questions provided to generate agent memory.")
            return None

        # Create a prompt that asks for answers to the questions
        prompt = "You are an AI assistant. Create a unique persona and answer the following interview questions from their perspective:\n\n"
        for q in self.questions:
            prompt += f"Interviewer: {q}\nAgent:\n\n"

        try:
            logging.info("Generating new agent memory...")
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative AI assistant that generates detailed and unique personas."},
                    {"role": "user", "content": prompt}
                ],
                max_retries=2,
            )
            
            # Extract the answers from the response
            answers = response.choices[0].message.content.strip().split("\n\n")
            
            # Combine questions and answers into nodes
            nodes = []
            for i, (question, answer) in enumerate(zip(self.questions, answers)):
                nodes.append(Node(
                    node_id=i,
                    node_type="observation",
                    content=f"Interviewer: {question}\n\n{answer}",
                    importance=80,  # Placeholder importance
                    created=0,      # Placeholder timestamp
                    last_retrieved=0 # Placeholder timestamp
                ))
            
            return AgentMemory(nodes=nodes)
        except Exception as e:
            logging.error(f"An error occurred while generating agent memory: {e}")
            return None

    def save_agent_to_file(self, agent_memory: AgentMemory):
        """Saves the generated agent memory to a file."""
        agent_id = str(uuid.uuid4())
        agent_dir = os.path.join(self.output_dir, agent_id, 'memory_stream')
        os.makedirs(agent_dir, exist_ok=True)
        
        file_path = os.path.join(agent_dir, 'nodes.json')
        
        nodes_data = [node.model_dump() for node in agent_memory.nodes]
        
        with open(file_path, 'w') as f:
            json.dump(nodes_data, f, indent=2)
        
        logging.info(f"Saved new agent to {file_path}")

    def create_population(self, size: int):
        """Creates a population of agents."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created output directory: {self.output_dir}")

        for i in range(size):
            logging.info(f"--- Generating Agent {i + 1}/{size} ---")
            agent_memory = self.generate_agent_memory()
            
            if agent_memory:
                self.save_agent_to_file(agent_memory)
            else:
                logging.warning(f"Skipping agent {i + 1} due to generation failure.")
