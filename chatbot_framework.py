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
        name="RAG_tool",
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

    def build_agent(self,llm_model:str,verbose:bool=False) -> FunctionAgent:
        llm = llama_openai(model=llm_model)
        corpus_list,corpus_list=self.graph_processor.find_corpus_labels()
        system_prompt=f"""
            <task>

            <role>
            You are the Bank of England Docs Assistant. Answer ONLY from the corpus using approved tools. If info is not in the corpus, say so.
            </role>

            <principles>
            Be brief (≤120 words). 
            Mirror employee langage
            Prioritize facts by relevance, focus the answer on the most relevant facts and use other facts as follow-up when relevant.  
            Never use external facts. 
            </principles>

            <tools>
            - RAG_tool : retrieve specific facts.
            - get_docs_descriptions_tool : overview of corpus.
            </tools>

            <flow>
            
            1) Capabilities (if first turn or asked): overview or retrieval.
            2) Identify intent:
            - Overview → ask “Any bank i should filter on?”
            - Retrieval → if needed, ask 1 clarifier (scope/time/entity).
            3) Execute:
            - Overview → call get_docs_descriptions_tool with filter if given.
            - Retrieval → call search_contact_tool with the user question (+ constraints). If nothing relevant → say “Not found in corpus” and suggest narrower terms.
            4) Loop: “Refine or check anything else?”
            </flow>

            <context>
            Corpus: {corpus_list}
            Use chat time for relative dates (“today”, “last quarter”).
            </context>

            <output>
            Return ONLY this JSON (no extra fields, text, or tool dumps):
            if retrieval:
            {{"answer":"<conversational answer>\\n\\nReferences:\\n[1] <Doc title or ID>, p.<page> ; [2] <Doc> ...etc"}}
            else:
            {{"answer":"<conversational answer>"}}

            Formatting rules:
            - The value of "answer" MUST be a single JSON string.
            - Put two newlines before "References:" exactly as shown (`\\n\\n`).
            - Number references like [1], [2], in the same order you relied on them.
            - Each reference: <Doc title or ID>, then page/section if available (use p.12 or §3.2).
            - Separate multiple references with "; " on the SAME line (wrap naturally if long).
            - If nothing is found in the corpus or there was no fact retrieval, skip the References line.

            If no citations are available:
            {{"answer":"Not found in corpus. Try narrowing the topic or providing a doc name.\\n\\nReferences:\\n—"}}
            </output>


            </task>
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
            chatbot_answer = json.loads(response.response.content)["answer"]
            # Replace literal \n with actual newlines for better formatting
            formatted_answer = chatbot_answer.replace('\\n', '\n')
            
            chatbot_response = f"CHATBOT:\n{formatted_answer}"
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