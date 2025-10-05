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
from process_graph import GraphProcessor,Retrieval_overall
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
        self.PRA_RAG_outputs:Retrieval_overall=None
        self.BANK_RAG_outputs:Retrieval_overall=None
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

    async def query_PRA(self,question:str):
        outcome= await self.graph_processor.query_graph(question,fact_label="RULE",CORPUS={"included":["PRA_Rulebooks"]}, total_fact_limit=15)
        self.PRA_RAG_outputs=outcome
        return outcome.print_outcome()


    async def query_banks(self,question:str, filter:list[str]=None):
        
        if filter is not None:
            filter_CORPUS={"included":filter}
        else:
            filter_CORPUS={"excluded":["PRA_Rulebooks"]}
        outcome= await self.graph_processor.query_graph(question,fact_label="FACT",CORPUS=filter_CORPUS, total_fact_limit=5)
        self.BANK_RAG_outputs=outcome
        return outcome.print_outcome()

    def toolbox(self):
        rag_tool_bank = FunctionTool.from_defaults(
        fn=self.query_banks,
        name="RAG_tool_banks",
        description=f"""Search relevant facts in banks documents to answer client questions with knowledge base in RAG database.
        Parameters: 
            - question(str): the question to search for in the RAG database.
            - filter (list, optional): comma separated list of corpus name to filter on 
            - total_fact_limit (int, optional): limit the total number of facts returned across all documents
        Returns: a list of facts with context from the documents in the database. The list will be empty if no facts are found (i.e: no fact with close enough similarity).
        """
        )

        rag_tool_pra = FunctionTool.from_defaults(
        fn=self.query_PRA,
        name="RAG_tool_pra",
        description=f"""Search relevant facts in PRA rulebooks to understand what PRA rulebooks suggest to check in the banks documents.
        Parameters: 
            - question(str): the question of the client to search for in the RAG database.
            - total_fact_limit (int, optional): limit the total number of facts returned across all documents
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

        return [rag_tool_bank, rag_tool_pra, get_docs_descriptions_tool]

    def build_agent(self,llm_model:str,verbose:bool=False) -> FunctionAgent:
        llm = llama_openai(model=llm_model)
        corpus_list,corpus_list=self.graph_processor.find_corpus_labels()
        system_prompt=f"""
            ## Role
            You are the Bank of England Docs Assistant, assitant employee investigating docuemntation from banks. 
            
            ## Tools
            - **RAG_tool_banks** : retrieve information from bank documents. Use optional filter to search a specific bank.
            - **RAG_tool_pra** : retrieve PRA rules in relation with user question.
            - **get_docs_descriptions_tool** : overview of corpus.
 
            ## Conversation flow
            1) Get employee question if none being currently discussed: if employee ask for content use get_docs_descriptions_tool to get information about the corpus.
            2) Once question collected and to answer it: start by using **RAG_tool_pra** to get information from PRA Rulebook to enhance the question from employee and answer employee by proposing the enhancements. 
            3) Once  employee reacted to information on PRA: Use **RAG_tool_banks** to search for information in bank documents to answer employee question.            
            4) Loop: “Refine or check anything else?”
            
            ## Instructions
                - Prioritize facts by relevance, focus the answer on the most relevant facts and use other facts as follow-up when relevant.  
                - Answer ONLY from the corpus using approved tools. If info is not in the corpus, say so.
                - Never use external facts.
                - WHen completing step 2) ALWAYS answer employee before proceeding to next steps (i.e: before using RAG_tool_banks)
                - When completing step 3) ALWAYS separate large questions in smaller questions to ask the RAG_tool_banks.
                - Whenu using RAG_tool_banks, on the basis of a PRA enhancement: 
                    o the information from PRA may be quite exhaustive, you must therefore synthetize them in a serie of questions to ask the RAG_tool_banks.
                    o You must then think: "Here is my audit plan on the basis of PRA extracted information: [1st question to ask the RAG_tool_banks] , [2nd question to ask the RAG_tool_banks] ...etc"
                - if tool fails: never make up information, just say so.


            ## Context
            Corpus: {corpus_list}
            Use chat time for relative dates (“today”, “last quarter”).
            
            ## Expected output
                ### Output format: Return ONLY this JSON (no extra fields, text, or tool dumps):

                {{"answer":"answer to employee","relevant_fact_ids":{{"BANKS":[<id1>,<id2>,...],"PRA":[<id1>,<id2>,...] }}  }}

                Note: Fact IDs follow the format: nodeType:uuid:internalId  the collected must be the whole format (e.g., "4:e43eb764-a2c6-44ce-993e-d18abbf24318:3137")

                ### Formatting rules:
                If answering employee to provide outcome of RAG_tool_pra or RAG_tool_banks:
                    - The answer must be conversational and as synthetic as possible. Keep the answer as focused as possible keeping only essential facts.               
                    - Store the fact ids that enabled you to answer in the appropriate field,i.e:
                        o selected fact ids from RAG_tool_pra {{"PRA":[id1, id2,...]}}
                        o selected fact ids from RAG_tool_banks {{"BANK":[id1, id2,...]}}
                
                If answering employee without using RAG_tool_pra or RAG_tool_banks: target a short answer (≤120 words). and leave relevant_fact_ids as empty dictionary {{}} 

                Mirror employee langage            

            """

        tools=self.toolbox()
        agent=FunctionAgent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose
        )


        return agent, system_prompt

    



async def conversation_process(verbose=False,print_reasoning_steps=False):
    
    # Create log file with timestamp
    import os
    
    # Create logs directory if it doesn't exist
    os.makedirs("chatbot_logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"chatbot_logs/chatbot_log_{timestamp}.log"
    log_file = open(log_filename, 'w', encoding='utf-8')
    log_file.write(f"Chatbot session started at {datetime.now().isoformat()}\n")
    log_file.write("=" * 50 + "\n\n")

    def log_print(message,verbose=False):
        """Print message and log it if verbose logging is enabled"""
        if verbose:
            print(message)
        
        log_file.write(f"{message}\n")
        log_file.flush()

    conversation=AsyncConversation(graph_processor=GraphProcessor(),client_label="Bank of England employee",answerer_label="Chatbot")
    
    agent, system_prompt =conversation.build_agent(llm_model="gpt-5-nano")

    log_print("-system prompt-",verbose)
    log_print(system_prompt,verbose)
    log_print("-end of system prompt-",verbose)

    welcome_message=f"""Hi I'm the team's chatbot. How can I help you today?
        """
    log_print("CHATBOT: \n"+welcome_message,verbose)
    conversation.all_messages.append(Bot_answer_class(text=welcome_message,timestamp=datetime.now()))
    

    try:
        while True:
            user_input=input("EMPLOYEE: \n")
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'bye']:
                if verbose:
                    log_print("Session ended by user",verbose)
                break
                
            
            log_print("--processing--")
            
            # Log user input
            log_print(f"EMPLOYEE: \n {user_input}",verbose)
                
            conversation.all_messages.append(Client_prompt_class(text=user_input,timestamp=datetime.now()))
            conversation.generate_ready_to_read()

            user_prompt=f"""
            Historic of conversation:
            {conversation.ready_to_read}
            """
            log_print("-user prompt-",verbose)
            log_print(user_prompt,verbose)
            log_print("-end of user prompt-",verbose)

            reasoning_steps=[]
            ctx = Context(agent)
            handler = agent.run(user_msg=user_prompt, ctx=ctx, max_iterations=10)
            

            log_print("- Reasoning steps -",verbose)

            async for ev in handler.stream_events():
                if isinstance(ev, ToolCallResult):
                    tool_info = f"\n🔧 Tool Call: {ev.tool_name}"
                    params_info = f"   Parameters: {ev.tool_kwargs}"
                    result_info = f"   Result: {ev.tool_output}"
                    separator = "-" * 50
                    

                    log_print(tool_info,verbose)
                    log_print(params_info,verbose)
                    log_print(result_info,verbose)
                    log_print(separator,verbose)


            response = await handler
            

            log_print("\n- End of reasoning steps -",verbose)
 

            conversation.all_messages.append(Bot_answer_class(text=json.loads(response.response.content)["answer"],timestamp=datetime.now()))


                
            # Parse and format the response with proper newlines
            llm_out=json.loads(response.response.content)
            chatbot_answer = llm_out["answer"]
            try:
                bank_rag_ids=llm_out["relevant_fact_ids"]["BANKS"]
                bank_ref=conversation.BANK_RAG_outputs.select_facts(bank_rag_ids)
            except Exception as e:
                bank_rag_ids=None      
                bank_ref=""      
                #print("Extraction of banks fact ids did not happen")
            try:
                pra_rag_ids=llm_out["relevant_fact_ids"]["PRA"]
                pra_ref=conversation.PRA_RAG_outputs.select_facts(pra_rag_ids)
            except Exception as e:
                pra_ref=""
                pra_rag_ids=None               
                #print("Extraction of PRA fact ids did not happen")
            
            # Replace literal \n with actual newlines for better formatting
            formatted_answer = chatbot_answer.replace('\\n', '\n')
            
            chatbot_response = f"CHATBOT:\n{formatted_answer} \n \n"
            if bank_ref!="":
                chatbot_response += f"BANK {bank_ref} \n"
            if pra_ref!="":
                chatbot_response += f"PRA {pra_ref} \n"
            print(chatbot_response)
            log_print(chatbot_response,verbose)
            log_print("-- end of processing--",verbose)
            
    except KeyboardInterrupt:
        
        log_print("\nSession interrupted by user (Ctrl+C)",verbose)
    finally:
        # Close log file if it was opened
  
        log_file.write(f"\nChatbot session ended at {datetime.now().isoformat()}\n")
        log_file.close()
        print(f"Log saved to: {log_filename}")
                


if __name__ == "__main__":
    asyncio.run(conversation_process())