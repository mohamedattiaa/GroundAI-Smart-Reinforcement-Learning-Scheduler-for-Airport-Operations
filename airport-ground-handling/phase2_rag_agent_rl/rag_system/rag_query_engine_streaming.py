"""
RAG Query Engine with Streaming Support
"""

from typing import List, Dict, Optional, Generator
from .retriever import ScenarioRetriever
import requests
import json


class RAGQueryEngineStreaming:
    """RAG system with streaming LLM responses"""
    
    def __init__(
        self,
        retriever: ScenarioRetriever,
        llm_url: str = "http://localhost:11434/api/generate",
        model_name: str = "llama3:8b"
    ):
        self.retriever = retriever
        self.llm_url = llm_url
        self.model_name = model_name
        
        print(f"✅ RAG Query Engine (Streaming) initialized")
        print(f"   LLM: {model_name}")
    
    def query_stream(
        self,
        question: str,
        n_retrieval: int = 3
    ) -> Generator[str, None, None]:
        """
        Answer question using RAG with streaming
        
        Args:
            question: User question
            n_retrieval: Number of scenarios to retrieve
        
        Yields:
            Chunks of the answer as they're generated
        """
        print(f"\n🔍 Processing query: {question}")
        
        # Retrieve scenarios
        print(f"   Retrieving {n_retrieval} similar scenarios...")
        retrieval_results = self.retriever.retrieve(question, n_results=n_retrieval)
        
        # Build prompt
        prompt = self._build_prompt(question, retrieval_results, True)
        
        # Stream response
        print(f"   Generating answer with {self.model_name}...")
        print("\n💡 Answer:")
        print("-" * 60)
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500,
                    "num_ctx": 2048
                }
            }
            
            response = requests.post(
                self.llm_url,
                json=payload,
                stream=True,
                timeout=10
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            text = chunk['response']
                            print(text, end='', flush=True)
                            yield text
                        
                        if chunk.get('done', False):
                            print("\n" + "-" * 60)
                            break
            else:
                error_msg = f"⚠️ Error: Status {response.status_code}"
                print(error_msg)
                yield error_msg
        
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            print(error_msg)
            yield error_msg
    
    def _build_prompt(self, question: str, retrieval_results: Dict, include_context: bool) -> str:
        """Build prompt for LLM"""
        
        prompt_parts = []
        
        # Keep prompt concise
        prompt_parts.append(
            "You are an airport operations expert. Answer concisely based on the data below.\n"
        )
        
        # Add context (limited)
        if include_context and retrieval_results['results']:
            prompt_parts.append("Historical data:")
            
            for i, result in enumerate(retrieval_results['results'][:2], 1):  # Only top 2
                prompt_parts.append(f"\n{i}. {result['document']}")
        
        # Question
        prompt_parts.append(f"\n\nQuestion: {question}")
        prompt_parts.append("\nAnswer (be specific and concise):")
        
        return "\n".join(prompt_parts)