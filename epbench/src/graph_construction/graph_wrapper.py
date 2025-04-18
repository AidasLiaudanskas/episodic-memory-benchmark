import time
from pathlib import Path
import networkx as nx
import json

# Import necessary components from the benchmark and graph construction modules
from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from epbench.src.graph_construction.embedding import get_openai_embeddings
from epbench.src.graph_construction.extractor import extract_entities_relationships_llm
from epbench.src.graph_construction.builder import build_knowledge_graph
from epbench.src.io.io import book_dirpath_func # To determine where to save the graph

# Default model configurations (can be overridden)
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EXTRACTION_MODEL = "gpt-4o-mini-2024-07-18"

def create_and_save_graph(
    benchmark: BenchmarkGenerationWrapper,
    env_file: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    extraction_model: str = DEFAULT_EXTRACTION_MODEL,
    force_rebuild: bool = False
) -> Path:
    """
    Orchestrates the creation of the knowledge graph for a given benchmark 
    and saves it to the benchmark's directory.

    Args:
        benchmark: An initialized BenchmarkGenerationWrapper instance.
        env_file: Path to the .env file.
        embedding_model: Name of the OpenAI model for embeddings.
        extraction_model: Name of the OpenAI model for entity/relationship extraction.
        force_rebuild: If True, rebuilds the graph even if it already exists.

    Returns:
        The path to the saved GraphML file.
    """
    start_time = time.time()
    
    # Determine the directory where the benchmark data is stored
    book_dir_path = book_dirpath_func(
        benchmark.nb_chapters(), 
        benchmark.nb_tokens(), 
        benchmark.data_folder, 
        benchmark.prompt_parameters, 
        benchmark.model_parameters, 
        benchmark.book_parameters
    )
    
    # Define the path for the graph file
    graph_filename = f"knowledge_graph_{embedding_model}_{extraction_model}.graphml"
    graph_filepath = book_dir_path / graph_filename

    if graph_filepath.exists() and not force_rebuild:
        print(f"Knowledge graph already exists at: {graph_filepath}")
        print(f"  Use force_rebuild=True to regenerate.")
        return graph_filepath

    print(f"Starting knowledge graph construction for benchmark...")
    print(f"  Target save path: {graph_filepath}")

    # 1. Get Chapter Data
    # Use split_chapters which is a dict {chapter_idx: chapter_text}
    # Ensure consistent ordering by sorting keys
    chapter_indices = sorted(benchmark.split_chapters.keys())
    chapter_texts = [benchmark.split_chapters[idx] for idx in chapter_indices]
    print(f"  Loaded {len(chapter_texts)} chapters.")

    # 2. Get Embeddings
    print(f"\nStep 1: Generating embeddings using {embedding_model}...")
    chapter_embeddings = get_openai_embeddings(
        texts=chapter_texts, 
        env_file=env_file,
        embedding_model=embedding_model
        # Add batch size if needed: embedding_batch_size=1024 
    )
    if len(chapter_embeddings) != len(chapter_texts):
         raise RuntimeError("Number of embeddings does not match number of chapters.")
    print(f"  Generated {len(chapter_embeddings)} embeddings.")

    # 3. Extract Entities and Relationships
    print(f"\nStep 2: Extracting entities/relationships using {extraction_model}...")
    extracted_data = extract_entities_relationships_llm(
        text_chunks=chapter_texts,
        env_file=env_file,
        model_name=extraction_model
        # Pass other relevant parameters like max_retries, request_timeout if needed
        # max_retries=3, 
        # request_timeout=120
    )
    if len(extracted_data) != len(chapter_texts):
        raise RuntimeError("Number of extracted data tuples does not match number of chapters.")
    print(f"  Extracted data for {len(extracted_data)} chapters.")

    # 4. Build Knowledge Graph
    print("\nStep 3: Building knowledge graph components...")
    # Define the output directory for the graph components
    graph_components_output_dir = book_dir_path 

    run_context = { # Example context, can be expanded
        "benchmark_prompt_parameters": benchmark.prompt_parameters,
        "benchmark_model_parameters": benchmark.model_parameters,
        "benchmark_book_parameters": benchmark.book_parameters,
        "embedding_model": embedding_model,
        "extraction_model": extraction_model,
    }

    stats = build_knowledge_graph(
        texts=chapter_texts, # Corrected keyword argument
        embeddings=chapter_embeddings,
        extracted_data=extracted_data,
        output_dir=graph_components_output_dir, # Pass the output directory
        run_context=run_context # Pass context info
    )
    print(f"  Graph components generated in {graph_components_output_dir}")
    if stats:
        print(f"  Stats: {stats}")

    # 5. Save Knowledge Graph (REMOVED - Saving is handled by build_knowledge_graph now)
    # print(f"\nStep 4: Saving knowledge graph to {graph_filepath}...") 
    # try:
    #     # Ensure the directory exists
    #     book_dir_path.mkdir(parents=True, exist_ok=True)
    #     # Use GraphML format for good attribute preservation
    #     nx.write_graphml(knowledge_graph, graph_filepath) # knowledge_graph is no longer returned
    #     print("  Graph saved successfully.")
    # except Exception as e:
    #     print(f"Error saving graph to {graph_filepath}: {e}")
    #     raise # Re-raise error to indicate failure
        
    end_time = time.time()
    print(f"\nKnowledge graph component generation finished in {end_time - start_time:.2f} seconds.")

    # Return the directory containing the graph components instead of a single file path
    return graph_components_output_dir

# Example of how this wrapper might be called (e.g., from quickstart.py)
if __name__ == '__main__':
    print("Running graph wrapper example...")
    print("  This example requires a pre-initialized BenchmarkGenerationWrapper instance.")
    print("  It simulates the process but doesn't run a full benchmark.")

    # --- Simulation Setup --- 
    # In a real scenario, these would come from your experiment script (e.g., quickstart.py)
    DUMMY_PROMPT_PARAMS = {'nb_events': 3, 'name_universe': 'test', 'name_styles': 'test', 'seed': 0, 'distribution_events': {'name': 'uniform'}}
    DUMMY_MODEL_PARAMS = {'model_name': 'test_generator', 'max_new_tokens': 10, 'itermax': 1}
    DUMMY_BOOK_PARAMS = {'indexing': 'test'}
    DUMMY_DATA_FOLDER = Path("./epbench/data") # Adjust path if needed
    DUMMY_ENV_FILE = "../../.env" # Adjust path if needed
    DUMMY_CHAPTERS = {1: "Alice went to the library.", 2: "Bob met Alice."} # Dummy chapters
    DUMMY_NB_TOKENS = 15

    # Simulate a BenchmarkGenerationWrapper instance with minimal necessary attributes
    class MockBenchmark:
        def __init__(self):
            self.prompt_parameters = DUMMY_PROMPT_PARAMS
            self.model_parameters = DUMMY_MODEL_PARAMS
            self.book_parameters = DUMMY_BOOK_PARAMS
            self.data_folder = DUMMY_DATA_FOLDER
            self.env_file = DUMMY_ENV_FILE
            self.split_chapters = DUMMY_CHAPTERS

        def nb_chapters(self): return len(self.split_chapters)
        def nb_tokens(self): return DUMMY_NB_TOKENS
    # --- End Simulation Setup ---

    mock_benchmark_instance = MockBenchmark()
    
    print(f"Simulating graph creation for a benchmark with {mock_benchmark_instance.nb_chapters()} chapters.")
    
    try:
        # Note: This will still call the actual OpenAI API for embeddings/extraction!
        # Replace with mock functions if you want to test without API calls.
        graph_path = create_and_save_graph(
            benchmark=mock_benchmark_instance, 
            env_file=DUMMY_ENV_FILE,
            # Use potentially faster/cheaper models for testing:
            # embedding_model="text-embedding-3-small", 
            # extraction_model="gpt-3.5-turbo",
            force_rebuild=True # Force it to run for the example
        )
        print(f"\nExample finished. Graph components saved to directory: {graph_path}") # Updated print message
        
        # Optional: Verify the output files exist
        # expected_files = [...] # List expected files
        # for fname in expected_files:
        #     fpath = graph_path / fname
        #     print(f"  Checking for {fname}: {'Found' if fpath.exists() else 'Missing'}")

    except FileNotFoundError as fnf_error:
         print(f"\nError: Could not find .env file at '{DUMMY_ENV_FILE}'. {fnf_error}")
         print("Please ensure the path in the example section is correct or create the file.")
    except ImportError as imp_error:
        print(f"\nError: Missing import. Ensure all dependencies are installed. {imp_error}")
    except Exception as e:
        print(f"\nAn error occurred during the wrapper example: {e}")
        print(traceback.format_exc()) 