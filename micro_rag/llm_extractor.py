import json
import logging
from langchain_ollama import ChatOllama
from micro_rag.prompts import prompt

# Set up simple logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class GraphExtractor:
    """
    Extracts entities and relationships from review texts using local Ollama model.
    """
    def __init__(self, model_name: str = "phi3.5", temperature: float = 0.0):
        logger.info(f"Initializing ChatOllama with model={model_name}, temp={temperature}")
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            format="json"
        )
        self.chain = prompt | self.llm

    def extract_from_text(self, review_text: str) -> dict:
        """
        Extracts structured graph data (entities, relations) from review text.
        Handles JSON parse failures gracefully by returning an empty graph structure.
        """
        logger.info(f"Invoking LLM for review text: {review_text[:60]}...")
        try:
            response = self.chain.invoke({"review_text": review_text})
            raw_content = response.content
            
            # Parse response as JSON
            graph_data = json.loads(raw_content)
            
            # Ensure keys exist
            if "entities" not in graph_data:
                graph_data["entities"] = []
            if "relations" not in graph_data:
                graph_data["relations"] = []
                
            return graph_data
            
        except json.JSONDecodeError as jde:
            logger.error(f"JSON Parsing Error: failed to parse LLM response. Response raw content:\n{raw_content}")
            logger.error(f"Exception details: {jde}")
            # Return empty structure to allow runner to skip/process gracefully
            return {"entities": [], "relations": []}
            
        except Exception as e:
            logger.error(f"Unexpected Error during LLM extraction: {e}")
            return {"entities": [], "relations": []}
