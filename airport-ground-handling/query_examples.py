"""
Example queries for RAG system
"""

def get_example_queries():
    """Get list of example queries to try"""
    
    return {
        "Resource Planning": [
            "How many fuel trucks do I need for 50 flights in 2 hours?",
            "What is the optimal size for a baggage loader fleet?",
            "How many catering trucks for 30 wide-body aircraft?",
            "What vehicle capacity is needed during peak hours?",
        ],
        
        "Delay Analysis": [
            "What causes the most delays in wide-body operations?",
            "How do equipment failures impact turnaround time?",
            "What is the average delay during peak hours?",
            "How to reduce delays for narrow-body aircraft?",
        ],
        
        "Optimization": [
            "How to optimize turnaround for A320 aircraft?",
            "Best practices for handling multiple aircraft simultaneously?",
            "What is the most efficient task sequencing?",
            "How to minimize vehicle idle time?",
        ],
        
        "Contingency Planning": [
            "How to handle equipment failures during peak hours?",
            "What is the backup plan for delayed arrivals?",
            "How to manage operations during bad weather?",
            "What contingency for simultaneous wide-body arrivals?",
        ],
        
        "Scheduling": [
            "What is the optimal scheduling for 5 concurrent flights?",
            "How to schedule tasks to minimize total delay?",
            "Best gate assignment strategy for mixed fleet?",
            "How to balance workload across vehicle fleet?",
        ]
    }


def print_example_queries():
    """Print all example queries"""
    
    queries = get_example_queries()
    
    print("="*70)
    print("EXAMPLE QUERIES FOR RAG SYSTEM")
    print("="*70)
    
    for category, questions in queries.items():
        print(f"\n📋 {category}:")
        print("-"*70)
        
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
    
    print("\n" + "="*70)
    print("Copy any question and use with:")
    print("  python demo_rag.py --mode interactive")
    print("="*70)


if __name__ == "__main__":
    print_example_queries()