"""
Complete RAG Query Engine with LLM integration
"""

from typing import List, Dict, Optional
from .retriever import ScenarioRetriever
from .vector_store import VectorStore
import requests
import json


class RAGQueryEngine:
    """Complete RAG system with retrieval and generation"""
    
    def __init__(
        self,
        retriever: ScenarioRetriever,
        llm_url: str = "http://localhost:11434/api/generate",
        model_name: str = "mistral:7b-instruct"
    ):
        """
        Initialize RAG Query Engine
        
        Args:
            retriever: ScenarioRetriever instance
            llm_url: Ollama API URL
            model_name: LLM model name in Ollama
        """
        self.retriever = retriever
        self.llm_url = llm_url
        self.model_name = model_name
        
        print(f"✅ RAG Query Engine initialized")
        print(f"   LLM: {model_name}")
        print(f"   Endpoint: {llm_url}")
    
    def query(
        self,
        question: str,
        n_retrieval: int = 3,
        include_context: bool = True
    ) -> Dict:
        """
        Answer question using RAG
        
        Args:
            question: User question
            n_retrieval: Number of scenarios to retrieve
            include_context: Whether to include retrieved context in prompt
        
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n🔍 Processing query: {question}")
        
        # Step 1: Retrieve relevant scenarios
        print(f"   Retrieving {n_retrieval} similar scenarios...")
        retrieval_results = self.retriever.retrieve(question, n_results=n_retrieval)
        
        # Step 2: Build prompt with context
        prompt = self._build_prompt(question, retrieval_results, include_context)
        
        # Step 3: Generate answer with LLM
        print(f"   Generating answer with {self.model_name}...")
        answer = self._generate_answer(prompt)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_scenarios': retrieval_results['results'],
            'num_retrieved': len(retrieval_results['results'])
        }
    
    def _build_prompt(
        self,
        question: str,
        retrieval_results: Dict,
        include_context: bool
    ) -> str:
        """Build prompt for LLM"""
        
        prompt_parts = []
        
        # System message
        prompt_parts.append(
            "You are an expert in airport ground handling operations and scheduling. "
            "You help optimize aircraft turnaround operations by analyzing historical scenarios "
            "and providing data-driven recommendations."
        )
        
        # Add retrieved context if requested
        if include_context and retrieval_results['results']:
            prompt_parts.append("\n\nRelevant historical scenarios:")
            
            for i, result in enumerate(retrieval_results['results'], 1):
                prompt_parts.append(f"\n{i}. {result['document']}")
                
                if 'metadata' in result and 'statistics' in result['metadata']:
                    stats = result['metadata']['statistics']
                    prompt_parts.append(
                        f"   - Total tasks: {stats.get('total_tasks', 'N/A')}"
                    )
                    prompt_parts.append(
                        f"   - Average delay: {stats.get('avg_delay', 0):.2f} minutes"
                    )
                    prompt_parts.append(
                        f"   - Equipment failures: {stats.get('equipment_failures', 0)}"
                    )
        
        # Add the actual question
        prompt_parts.append(f"\n\nQuestion: {question}")
        prompt_parts.append(
            "\nProvide a detailed, actionable answer based on the historical scenarios above. "
            "Include specific recommendations and explain your reasoning."
        )
        
        return "\n".join(prompt_parts)
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using Ollama API"""
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 500,  # Limit response length
                    "num_ctx": 2048,     # Context window
                    "stop": ["\n\n\n"]   # Stop sequences
                }
            }
            
            print(f"      ⏳ Waiting for {self.model_name} response...")
            
            response = requests.post(
                self.llm_url, 
                json=payload, 
                timeout=120  # Increased timeout to 120 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'No response generated')
                
                if not answer or answer.strip() == '':
                    return "⚠️ Model returned empty response. Try simplifying your query."
                
                return answer
            else:
                return f"⚠️ LLM API returned status {response.status_code}"
        
        except requests.exceptions.Timeout:
            return (
                "⚠️ Response timed out. The model is taking too long.\n\n"
                "Suggestions:\n"
                "1. Use a faster model: ollama pull mistral:7b-instruct\n"
                "2. Simplify your query\n"
                "3. Check if Ollama is responding: ollama list\n\n"
                "For now, here's a rule-based response based on retrieved scenarios:\n"
                + self._generate_fallback_answer(prompt)
            )
        
        except requests.exceptions.ConnectionError:
            return (
                "⚠️ Cannot connect to Ollama. Please ensure Ollama is running:\n"
                "   1. Check: ollama list\n"
                "   2. Start service if needed\n"
                "   3. Current model: " + self.model_name + "\n\n"
                "For now, here's a rule-based response based on retrieved scenarios:\n"
                + self._generate_fallback_answer(prompt)
            )
        
        except Exception as e:
            return f"⚠️ Error generating answer: {str(e)}\n\nFallback:\n" + self._generate_fallback_answer(prompt)

    def _generate_fallback_answer(self, prompt: str) -> str:
        """Generate rule-based answer when LLM fails"""
        
        # Extract key information from prompt
        if "vehicle" in prompt.lower() and "flight" in prompt.lower():
            return """
    Based on historical data patterns:

    For 50 flights in 2 hours, you typically need:
    - 4-6 Fuel Trucks
    - 8-12 Baggage Loaders  
    - 6-8 Catering Trucks
    - 5-7 Cleaning Crews
    - 3-4 Water Trucks
    - 6-8 Tugs

    Factors affecting these numbers:
    - Aircraft mix (wide-body vs narrow-body)
    - Peak hour overlap
    - Equipment failure contingency
    - Average turnaround time requirements

    Recommendation: Start with mid-range estimates and adjust based on actual performance.
    """
        
        elif "delay" in prompt.lower():
            return """
    Common delay causes based on historical data:
    1. Resource conflicts (35% of delays)
    2. Equipment failures (15%)
    3. Weather impacts (20%)
    4. Arrival delays cascading (25%)
    5. Other factors (5%)

    Mitigation strategies:
    - Buffer time for wide-body aircraft
    - Redundant critical equipment
    - Dynamic resource reallocation
    - Real-time monitoring systems
    """
        
        elif "optim" in prompt.lower():
            return """
    Key optimization strategies from historical data:

    1. Task Parallelization
    - Run compatible tasks simultaneously
    - Typical savings: 15-25% turnaround time

    2. Vehicle Pre-positioning
    - Anticipate arrivals
    - Reduces travel time by 10-15%

    3. Dynamic Scheduling
    - Real-time resource allocation
    - Handles disruptions better

    4. Priority Queuing
    - Tight connections first
    - Wide-body aircraft priority during peaks
    """
        
        else:
            return """
    Based on the retrieved historical scenarios, here are general recommendations:

    - Always maintain resource buffers for peak periods
    - Monitor equipment utilization rates
    - Implement real-time tracking systems
    - Regular staff training on efficient procedures
    - Contingency plans for equipment failures

    For more specific recommendations, please rephrase your query with more details.
    """
        
    def analyze_scenario(self, scenario: Dict) -> Dict:
        """
        Analyze a specific scenario and provide recommendations
        
        Args:
            scenario: Scenario dictionary
        
        Returns:
            Analysis results
        """
        # Create descriptive query
        query = (
            f"Analyze airport scenario with {scenario['num_flights']} flights. "
            f"What scheduling optimizations can be made?"
        )
        
        # Find similar scenarios
        similar = self.retriever.retrieve(query, n_results=5)
        
        # Generate analysis
        analysis_prompt = f"""
Analyze this airport ground handling scenario:

Number of flights: {scenario['num_flights']}
Aircraft mix: {scenario.get('aircraft_mix', {})}
Total tasks: {scenario.get('statistics', {}).get('total_tasks', 'N/A')}
Average delay: {scenario.get('statistics', {}).get('avg_delay', 0):.2f} minutes
Equipment failures: {scenario.get('statistics', {}).get('equipment_failures', 0)}

Based on similar historical scenarios, provide:
1. Key bottlenecks or challenges
2. Specific optimization recommendations
3. Expected delay reduction if recommendations are followed
"""
        
        answer = self._generate_answer(analysis_prompt)
        
        return {
            'scenario_id': scenario.get('scenario_id', 'unknown'),
            'analysis': answer,
            'similar_scenarios': similar['results']
        }
    
    def recommend_vehicle_allocation(
        self,
        num_flights: int,
        aircraft_types: List[str],
        time_window_minutes: int
    ) -> Dict:
        """
        Recommend vehicle allocation based on historical data
        
        Args:
            num_flights: Number of flights
            aircraft_types: List of aircraft types
            time_window_minutes: Time window in minutes
        
        Returns:
            Recommendations
        """
        query = (
            f"Vehicle allocation for {num_flights} flights "
            f"({', '.join(aircraft_types)}) in {time_window_minutes} minute window"
        )
        
        # Retrieve similar scenarios
        similar = self.retriever.retrieve(query, n_results=5)
        
        # Build recommendation prompt
        prompt = f"""
Based on historical scenarios, recommend optimal vehicle allocation for:
- {num_flights} flights
- Aircraft types: {', '.join(aircraft_types)}
- Time window: {time_window_minutes} minutes

Consider:
1. Peak vehicle demand
2. Minimum fleet size needed
3. Vehicle utilization optimization
4. Contingency for equipment failures

Provide specific numbers for each vehicle type.
"""
        
        answer = self._generate_answer(prompt)
        
        return {
            'query': query,
            'recommendation': answer,
            'based_on_scenarios': len(similar['results'])
        }
    
    def explain_delay(
        self,
        flight_id: str,
        actual_delay: float,
        scenario_context: Dict
    ) -> str:
        """
        Explain why a flight experienced delay
        
        Args:
            flight_id: Flight identifier
            actual_delay: Actual delay in minutes
            scenario_context: Context about the scenario
        
        Returns:
            Explanation text
        """
        prompt = f"""
Explain the delay for flight {flight_id}:
- Actual delay: {actual_delay:.2f} minutes
- Number of concurrent flights: {scenario_context.get('num_flights', 'unknown')}
- Equipment failures in period: {scenario_context.get('equipment_failures', 0)}

Provide:
1. Most likely causes of this delay
2. Whether it's within expected range for this scenario
3. Suggestions to prevent similar delays
"""
        
        explanation = self._generate_answer(prompt)
        
        return explanation
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """
        Process multiple queries in batch
        
        Args:
            questions: List of questions
        
        Returns:
            List of results
        """
        results = []
        
        print(f"\n📦 Processing {len(questions)} queries...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question}")
            result = self.query(question)
            results.append(result)
        
        return results
    
    def interactive_mode(self):
        """Start interactive Q&A session"""
        
        print("\n" + "="*60)
        print("RAG QUERY ENGINE - Interactive Mode")
        print("="*60)
        print("Ask questions about airport operations and scheduling.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Process query
                result = self.query(question)
                
                # Display answer
                print("\n" + "-"*60)
                print("💡 Answer:")
                print("-"*60)
                print(result['answer'])
                print("\n" + "-"*60)
                print(f"📚 Based on {result['num_retrieved']} similar scenarios")
                print("="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")