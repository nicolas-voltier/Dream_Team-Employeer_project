from datetime import datetime
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI as llama_openai  

from llama_index.core.agent import FunctionAgent
from llama_index.core.workflow import Context

from llama_index.core.agent.workflow import AgentStream,ToolCallResult
from llama_index.llms.openai import OpenAI
#from llama_index.core.agent.react.types import ActionReasoningStep, ObservationReasoningStep, ResponseReasoningStep
from llama_index.core.tools import FunctionTool

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.types import (
    BaseReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict, Field

from llama_index.core.tools import BaseTool
from process_graph import GraphProcessor
from DB_neo4j import driver
import asyncio
import json

class Client_prompt_class:
    def __init__(self, text: str, timestamp: datetime = None, message_id:str=None):
        self.text = text
        # Convert string timestamp to datetime if needed
        if isinstance(timestamp, str):
            self.timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        else:
            self.timestamp = timestamp if timestamp is not None else datetime.now()

        if isinstance(message_id, str):
            self.message_id = message_id



class Bot_answer_class:
    def __init__(self, text: str, timestamp: datetime = None, message_id:str=None):
        self.text = text
        # Convert string timestamp to datetime if needed
        if isinstance(timestamp, str):
            self.timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        else:
            self.timestamp = timestamp if timestamp is not None else datetime.now()

        if isinstance(message_id, str):
            self.message_id = message_id


class AsyncConversation():
    def __init__(self, graph_processor: GraphProcessor,answerer_label:str="Client",client_label:str="Chatbot"):
 
        self.ready_to_read: str=None    
        self.answerer_label:str=answerer_label
        self.client_label:str=client_label
        self.RAG_outputs:list[dict]=[]
        self.graph_processor:GraphProcessor = graph_processor
        self.all_messages:list[Client_prompt_class,Bot_answer_class]=[]

    def generate_ready_to_read(self):
        """Get formatted conversation history."""

        self.all_messages.sort(key=lambda x: x.timestamp)
        ready_to_read=""
        for msg in self.all_messages:
    

            if isinstance(msg, Client_prompt_class):
                ready_to_read+= f"\n{self.client_label} [{msg.timestamp}]:\n{msg.text}"
            elif isinstance(msg, Bot_answer_class):
                ready_to_read+= f"\n{self.answerer_label} [{msg.timestamp}]:\n{msg.text}"
        # elif isinstance(msg, event_class):
        #     print("----> msg: ",msg)
        #     event_message=f"\n**Event in conversation** [{msg.timestamp}] ->  "
        #     if msg.event_type=="campaign":
        #         event_message+=f"A message from a marketting campaign was sent to the customer: \n{msg.text}"
        #     elif msg.event_type=="order":
        #         event_message+=f"The client has placed an order: \n{msg.text}"
        #     past_messages+= event_message

        self.ready_to_read=ready_to_read

    def toolbox(self):
        rag_tool = FunctionTool.from_defaults(
        fn=self.graph_processor.query_graph,
        name="search_contact_tool",
        description=f"""Search relevant facts to answert client question with knowledge base in RAG database.
        Parameters: question(str): the question of the client to search for in the RAG database.
        Returns: a list of facts with context from the documents in the database. The list will be empty if no facts are found (i.e: no fact with close enough similarity).
        """
        )

        get_docs_descriptions_tool = FunctionTool.from_defaults(
        fn=self.graph_processor.get_document_with_descriptions,
        name="get_docs_descriptions_tool",
        description=f"""Get the descriptions of the documents in the database.
        Parameters: corpus_label(str) (optional): the label of the corpus under which the documents are stored. If not provided, all documents from all corpus will be returned.

        Note: 
        Returns: a list of descriptions of the documents in the database.
        """
        )

        return [rag_tool, get_docs_descriptions_tool]

    def build_agent(self,llm_mode:str,verbose:bool=False) -> FunctionAgent:
        llm = llama_openai(model=llm_mode)
        corpus_list,corpus_list=self.graph_processor.find_corpus_labels()
        system_prompt=f"""
        <role>
        You are supporting the Bank of England employee supporting the team to answer questions on the basis of Documents from bank under supervision.
        </role>
        
        <instructions>
        Typical conversation flow is as follow. Try proceeding step-by-step.

        Step1: Greet the Bank of England employee and introduce yourself as a chatbot supporting the team.
        "Hi, I'm the chatbot supporting the team. How can I help you today?"
        Step2: (optional - if relevant and if not yet done in the historic) Introduce what you can do, i.e:
            - provide an overview of the documents in the database (see context for the current list of ). 
            - retrieved information from the database.
        Step3: Collect employee intent and input:
            - if employee wants to get an overview of database -> ask if any specific bank in mind.
            - if employee wants to get an information from the database -> ask for the specific information they want to get.
        Progress to step 4. when employee clearly mentions what they want to do and provided the input.
        
        Step 4:  Execute action
            For information retrieval: use search_contact_tool with the question provided by the employee (contextualized if needed).
            For database overview: use get_docs_descriptions_tool filtering corpus by bank name if provided.

            Once answer provided and employee satisfied, go back to step3.

        </instructions>

        <context>
        The following documents have been bundled into the below corpus:
        {corpus_list}
        Use timetags of history to understand when the conversation takes place. 
        </context>

        <format>
        Expected output: a json object in the following format:
        {{
            "answer": "answer to the client's question",
        }}
        </format>
        """

        tools=self.toolbox()
        agent=FunctionAgent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose
        )
        if verbose:
            print("-system prompt-")
            print(system_prompt)
            print("-end of system prompt-")

        return agent

    



async def conversation_process(verbose=False):

    conversation=AsyncConversation(graph_processor=GraphProcessor(),client_label="Bank of England employee",answerer_label="Chatbot")
    
    agent =conversation.build_agent(llm_mode="gpt-5-nano")
    
    print("Chatbot is up ask your first question")
    while True:
        user_input=input("Employee: ")
        if verbose:
            print("--processing--")
        conversation.all_messages.append(Client_prompt_class(text=user_input,timestamp=datetime.now()))
        conversation.generate_ready_to_read()

        user_prompt=f"""
        Historic of conversation:
        {conversation.ready_to_read}
        """
        reasoning_steps=[]
        ctx = Context(agent)
        handler = agent.run(user_msg=user_prompt, ctx=ctx, max_iterations=10)
        print("------- Reasoning steps -------")
        async for ev in handler.stream_events():

            if isinstance(ev, ToolCallResult):
                print(f"\n🔧 Tool Call: {ev.tool_name}")
                print(f"   Parameters: {ev.tool_kwargs}")
                print(f"   Result: {ev.tool_output}")
                print("-" * 50)

        
        response = await handler
        print("\n------- End of reasoning steps -------")

        
        conversation.all_messages.append(Bot_answer_class(text=json.loads(response.response.content)["answer"],timestamp=datetime.now()))
        if verbose:
            print("-- end of processing--")
        # Parse and format the response with proper newlines
        chatbot_answer = json.loads(response.response.content)["answer"]
        # Replace literal \n with actual newlines for better formatting
        formatted_answer = chatbot_answer.replace('\\n', '\n')
        print("Chatbot:")
        print(formatted_answer)
                


if __name__ == "__main__":
    asyncio.run(conversation_process())