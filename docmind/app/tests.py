# app/tests.py
import time
import pandas as pd
from app.ingestion import DocumentProcessor
from app.indexing import DocumentIndexer
from app.retrieval import LightRAGRetriever
from app.agents import DocMindAgent
from app.config import DATA_DIR
import os

def run_benchmark_tests():
    """Run benchmark tests for the DocMind system."""
    results = {
        'test_name': [],
        'retrieval_type': [],
        'execution_time': [],
        'result_quality': []
    }
    
    # Setup test environment
    processor = DocumentProcessor()
    default_docs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
                   if os.path.isfile(os.path.join(DATA_DIR, f))]
    nodes = processor.ingest_documents(file_paths=default_docs)
    
    indexer = DocumentIndexer()
    indexer.create_vector_index(nodes)
    indexer.create_knowledge_graph(nodes)
    
    retriever = LightRAGRetriever(indexer)
    retriever.setup_bm25_retriever(nodes)
    
    agent = DocMindAgent(indexer, retriever)
    
    # Test queries
    test_queries = [
        "What are the main components of a RAG system?",
        "How does LightRAG improve retrieval quality?",
        "Explain the difference between vector and graph-based search",
        "What tools does the DocMind agent have access to?",
        "How can I implement a hybrid search approach?"
    ]
    
    # Test different retrieval methods
    retrieval_types = ["hybrid", "graph", "transformed"]
    
    for query in test_queries:
        for r_type in retrieval_types:
            # Measure execution time
            start_time = time.time()
            response = agent.search_documents(query, retrieval_type=r_type)
            execution_time = time.time() - start_time
            
            # Evaluate quality (simplified - in production you'd use a more sophisticated metric)
            # Here we're using response length and number of sources as a proxy for quality
            source_count = len(response.get("source_nodes", []))
            response_length = len(response.get("response", ""))
            quality_score = min(10, (source_count * 2) + (response_length / 100))
            
            # Record results
            results['test_name'].append(query)
            results['retrieval_type'].append(r_type)
            results['execution_time'].append(round(execution_time, 3))
            results['result_quality'].append(round(quality_score, 2))
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n===== DocMind Performance Benchmark =====")
    print(f"Total tests run: {len(df)}")
    
    # Average metrics by retrieval type
    print("\n----- Performance by Retrieval Type -----")
    retrieval_summary = df.groupby('retrieval_type').agg({
        'execution_time': ['mean', 'min', 'max'],
        'result_quality': ['mean', 'min', 'max']
    })
    print(retrieval_summary)
    
    # Find best method for each query
    print("\n----- Best Method by Query -----")
    best_methods = df.loc[df.groupby('test_name')['result_quality'].idxmax()]
    print(best_methods[['test_name', 'retrieval_type', 'result_quality']])
    
    # Save results to CSV
    df.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to benchmark_results.csv")
    
    return df

def optimize_system_parameters():
    """Optimize system parameters based on test results."""
    # This function would tune parameters based on benchmark results
    # For production systems, you might use more sophisticated optimization
    
    # Example optimization: determine best retrieval_type based on benchmarks
    benchmark_df = run_benchmark_tests()
    
    # Calculate average quality and speed for each retrieval type
    retrieval_performance = benchmark_df.groupby('retrieval_type').agg({
        'execution_time': 'mean',
        'result_quality': 'mean'
    })
    
    # Normalize scores (lower time is better, higher quality is better)
    retrieval_performance['time_score'] = 1 - ((retrieval_performance['execution_time'] - 
                                            retrieval_performance['execution_time'].min()) / 
                                           (retrieval_performance['execution_time'].max() - 
                                            retrieval_performance['execution_time'].min()))
    
    retrieval_performance['quality_score'] = ((retrieval_performance['result_quality'] - 
                                            retrieval_performance['result_quality'].min()) / 
                                           (retrieval_performance['result_quality'].max() - 
                                            retrieval_performance['result_quality'].min()))
    
    # Calculate combined score (equal weight to speed and quality)
    retrieval_performance['combined_score'] = (retrieval_performance['time_score'] + 
                                             retrieval_performance['quality_score']) / 2
    
    # Find best overall method
    best_method = retrieval_performance['combined_score'].idxmax()
    
    print(f"\nBest overall retrieval method: {best_method}")
    print("\nOptimized parameters:")
    print(f"- DEFAULT_RETRIEVAL_TYPE = '{best_method}'")
    
    # Update config file with optimized settings
    with open('app/config.py', 'a') as f:a
        f.write(f"\n# Optimized settings based on benchmarks\n")
        f.write(f"DEFAULT_RETRIEVAL_TYPE = '{best_method}'\n")
    
    print("\nConfiguration updated with optimized settings.")
    
    return best_method

if __name__ == "__main__":
    print("Running DocMind system tests and optimization...")
    best_method = optimize_system_parameters()
    print(f"Testing and optimization complete. Recommended retrieval type: {best_method}")