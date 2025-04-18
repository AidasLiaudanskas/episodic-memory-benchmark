import sys
import os
from pathlib import Path
import networkx as nx
import json
import traceback
import pandas as pd # Add import for potential parquet loading

# --- Adjust Python Path --- 
# Add the project root to the Python path to allow imports like epbench.src. ...
# This assumes the script is run from the 'experiments' directory or the project root.
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.resolve() # Go up two levels from experiments/ to project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to sys.path")
# --------------------------

# Now imports should work if the environment is set up correctly
try:
    from epbench.src.graph_construction.graph_wrapper import create_and_save_graph, DEFAULT_EMBEDDING_MODEL, DEFAULT_EXTRACTION_MODEL
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure you are running this script from the 'experiments' directory or the project root,")
    print("and that the required packages (networkx, openai, pandas) are installed.")
    print(f"Project root detected as: {project_root}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Test Configuration --- 
# Define simple text input
TEST_STORY_CHAPTERS = {
    1: "Dr. Aris Thorne adjusted his spectacles, peering at the chrono-display. Project Chimera was nearing activation. The temporal core hummed softly in the adjacent chamber.",
    2: "Suddenly, red lights flashed. Alarms blared. \"Containment breach in Sector Gamma!\" echoed Lena Petrova, his lead assistant, her voice tight with panic.",
    3: "Thorne rushed towards Sector Gamma, grabbing a stabilization device from the wall rack. Petrova followed, relaying damage reports through her comm link.",
    4: "They found the temporal core emitting erratic energy pulses. The Chimera device flickered. \"It's destabilizing! We need to shut it down!\" Thorne yelled over the noise, working on the control panel."
}

# Define parameters needed for the mock benchmark and file paths
# These mimic the structure used in BenchmarkGenerationWrapper and book_dirpath_func
TEST_PROMPT_PARAMS = {'nb_events': len(TEST_STORY_CHAPTERS), 'name_universe': 'test_graph', 'name_styles': 'test', 'seed': 42, 'distribution_events': {'name': 'test'}}
TEST_MODEL_PARAMS = {'model_name': 'test_generator', 'max_new_tokens': 10, 'itermax': 1}
TEST_BOOK_PARAMS = {'indexing': 'test'}
TEST_DATA_FOLDER = project_root / "epbench" / "data" / "test_graph_output" # Save output in a dedicated test folder
TEST_ENV_FILE = project_root / ".env" # Assumes .env is in the project root
TEST_NB_TOKENS = sum(len(t) for t in TEST_STORY_CHAPTERS.values()) # Simple token count approximation

# Create a mock class that simulates the necessary parts of BenchmarkGenerationWrapper
class MockTestBenchmark:
    def __init__(self):
        self.prompt_parameters = TEST_PROMPT_PARAMS
        self.model_parameters = TEST_MODEL_PARAMS
        self.book_parameters = TEST_BOOK_PARAMS
        self.data_folder = TEST_DATA_FOLDER
        # self.env_file = TEST_ENV_FILE # env_file is passed directly to create_and_save_graph
        self.split_chapters = TEST_STORY_CHAPTERS

    def nb_chapters(self): return len(self.split_chapters)
    def nb_tokens(self): return TEST_NB_TOKENS

# --- Main Test Execution --- 
def run_test():
    print("--- Starting Independent Graph Creation Test ---")
    
    # Ensure the .env file exists
    if not TEST_ENV_FILE.is_file():
        print(f"Error: .env file not found at expected location: {TEST_ENV_FILE}")
        print("Please ensure the .env file exists in the project root or update TEST_ENV_FILE path.")
        return

    # Create the mock benchmark instance
    mock_benchmark = MockTestBenchmark()
    
    print(f"Using Mock Benchmark with {mock_benchmark.nb_chapters()} chapters.")
    print(f"Output will be saved in: {TEST_DATA_FOLDER}")
    # Ensure output directory exists (create_and_save_graph also does this, but good practice)
    TEST_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    try:
        # Call the main graph creation function
        # Using default models for this test, can be overridden
        print(f"\nCalling create_and_save_graph...")
        graph_path = create_and_save_graph(
            benchmark=mock_benchmark, 
            env_file=str(TEST_ENV_FILE),
            embedding_model=DEFAULT_EMBEDDING_MODEL, # Use defaults or specify test models
            extraction_model=DEFAULT_EXTRACTION_MODEL, 
            force_rebuild=True # Force creation for this test
        )
        print(f"\nGraph creation process completed.")
        # graph_path is now the directory where components are saved
        output_dir_path = graph_path 
        print(f"Graph components saved to directory: {output_dir_path}")

        # Validation Step: Check for expected output files and LanceDB directory
        if output_dir_path.is_dir(): # Check if it's a directory
            print("\nValidating saved GraphRAG components...")
            
            # Check for LanceDB directory first
            lancedb_dir_path = output_dir_path / "lancedb"
            lancedb_found = False
            if lancedb_dir_path.is_dir():
                 print(f"  [FOUND DIR] lancedb")
                 lancedb_found = True
                 # Optional: Add deeper checks, like trying to connect or list tables
                 # try:
                 #     import lancedb
                 #     db = lancedb.connect(lancedb_dir_path)
                 #     print(f"    - LanceDB tables: {db.table_names()}")
                 # except Exception as ldb_e:
                 #     print(f"    - Warn: Could not connect/inspect LanceDB: {ldb_e}")
            else:
                 print(f"  [MISSING DIR] lancedb (Check builder logs/implementation)")
                 # Consider if this should be a hard failure depending on requirements

            # Check for expected Parquet files
            expected_files = [
                "create_final_text_units.parquet", 
                "create_final_entities.parquet", # Placeholder in builder
                "create_final_nodes.parquet",
                "create_final_relationships.parquet",
                "create_final_communities.parquet", # Placeholder in builder
                "create_final_community_reports.parquet", # Placeholder in builder
                "create_final_covariates.parquet", # Placeholder in builder
                "stats.json", 
                "context.json"
            ]
            
            all_files_found = True
            missing_core_files = []
            for fname in expected_files:
                fpath = output_dir_path / fname
                is_placeholder_output = any(sub in fname for sub in ["entities.parquet", "communities.parquet", "reports.parquet", "covariates.parquet"])
                
                if fpath.exists():
                    print(f"  [FOUND FILE] {fname}")
                    # Optional: Add parquet/json loading checks here if needed
                else:
                    if is_placeholder_output:
                        print(f"  [MISSING/EMPTY] {fname} (Potentially due to placeholder implementation in builder.py)")
                        # Don't fail validation for missing placeholder outputs yet
                    else:
                        print(f"  [MISSING FILE] {fname} - ERROR!")
                        all_files_found = False
                        missing_core_files.append(fname)
            
            # Determine overall success
            # Modify criteria based on whether LanceDB/placeholders are strictly required
            validation_successful = all_files_found and lancedb_found # Example: require LanceDB and non-placeholder files
            
            if validation_successful:
                print("\nValidation successful: All expected core files and LanceDB directory found.")
            else:
                print("\nValidation failed:")
                if not lancedb_found:
                     print("  - LanceDB directory missing.")
                if missing_core_files:
                     print(f"  - Missing core files: {missing_core_files}")
                print("  (Note: Some missing files might be due to placeholder implementations in builder.py)")

            # Remove the old GraphML loading and inspection code
            # loaded_graph = nx.read_graphml(graph_path)
            # print(f"  Successfully loaded graph from {graph_path}")
            # ... (rest of old validation code removed) ...

        else:
            print(f"Error: Expected output directory was not created or is not a directory: {output_dir_path}")

    except FileNotFoundError as fnf_error:
         print(f"\nError during test: Could not find .env file. {fnf_error}")
    except ImportError as imp_error:
        print(f"\nError during test: Missing import. {imp_error}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        print(traceback.format_exc())
    finally:
        print("--- Test Finished ---")

if __name__ == "__main__":
    run_test() 