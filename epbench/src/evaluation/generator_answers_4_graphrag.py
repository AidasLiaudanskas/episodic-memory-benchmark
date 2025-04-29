#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate answers using a GraphRAG approach."""

import os
import time
import traceback
import asyncio # Added for running async API calls
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import threading # Added for print lock, similar to prompting script

# REMOVED: Specific logger configurations
# logging.getLogger("graphrag").setLevel(logging.WARNING)
# logging.getLogger("lancedb").setLevel(logging.WARNING)

# ADDED: Set global logging level to WARNING
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
# Removed: import networkx as nx
# Removed: import lancedb
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not found. Skipping .env file loading.")
    def load_dotenv(*args, **kwargs): pass # No-op function

# --- GraphRAG Imports using graphrag.api ---
from graphrag import api # Using the main API functions
from graphrag.config.load_config import load_config


# ADDED: Imports for saving and path generation
from epbench.src.io.io import answer_filepath_func, export_list, import_list
from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper # For type hint
# ADDED: Import SettingsWrapper to get MAX_WORKERS
from epbench.src.models.settings_wrapper import SettingsWrapper

# --- Constants ---
DEFAULT_ANSWERING_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small" # Keep for reference

# Create a lock for thread-safe printing (similar to prompting script)
global print_lock
print_lock = threading.Lock()

# --- Helper Function for Single Question Processing ---

async def process_single_question_graphrag(
    q_idx: int,
    question: str,
    # GraphRAG components (passed from main function)
    graphrag_config: Dict,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    community_reports: pd.DataFrame,
    text_units: pd.DataFrame,
    relationships: pd.DataFrame,
    covariates: pd.DataFrame,
    # Path and config parameters
    nb_chapters: int,
    nb_tokens: int,
    data_folder: Path,
    prompt_parameters: Dict,
    model_parameters: Dict,
    book_parameters: Dict,
    answering_parameters: Dict,
    community_level: int,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Processes a single question using GraphRAG's local_search API.
    Does NOT handle caching - assumes the question needs processing.
    Saves the result to a file.

    Returns:
        A dictionary containing 'q_idx', 'question', 'llm_answer', 'retrieved_context'.
    """
    output_filepath = answer_filepath_func(
        q=q_idx,
        nb_chapters=nb_chapters, nb_tokens=nb_tokens, data_folder=data_folder,
        prompt_parameters=prompt_parameters, model_parameters=model_parameters,
        book_parameters=book_parameters, answering_parameters=answering_parameters
    )

    answer = "Error generating answer via GraphRAG." # Default error answer
    retrieved_context = "Context unavailable due to error." # Default error context

    async with semaphore: # Limit concurrency
        with print_lock:
             print(f"    Starting GraphRAG query for Q_ID: {q_idx}...")
        try:
            api_response, api_context = await api.local_search(
                config=graphrag_config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=covariates,
                query=question,
                community_level=community_level,
                response_type="Multiple Paragraphs"
            )
            answer = api_response
            # Ensure context is string even if None or complex object
            retrieved_context = str(api_context) if api_context is not None else "Context Unavailable"

            with print_lock:
                 print(f"    Finished GraphRAG query for Q_ID: {q_idx}.")

            # Save successful result to cache
            answer_to_save = str(answer) if answer is not None else "Processing Skipped/Failed"
            try:
                output_filepath.parent.mkdir(parents=True, exist_ok=True)
                export_list(answer_to_save, output_filepath)
                # with print_lock:
                #     print(f"  Successfully saved answer to: {output_filepath}")
            except Exception as e_save:
                with print_lock:
                    print(f"  Error saving answer for q_idx {q_idx} to JSON: {e_save}")

        except Exception as e_async:
            with print_lock:
                print(f"    ERROR inside async local_search for Q_ID {q_idx} ('{question[:30]}...'): {e_async}")
            retrieved_context = f"Error during GraphRAG processing: {e_async}"

    # Return processed data including original row info
    processed_data = {
        'q_idx': q_idx,
        'question': question,
        'llm_answer': str(answer) if answer is not None else "Processing Skipped/Failed",
        'retrieved_context': str(retrieved_context) # Already ensured string above
    }
    return processed_data


# --- Main Function ---

def generate_answers_graphrag(
    df_questions: pd.DataFrame,
    graph_components_dir: Path, # This is the specific path to the graphrag index (e.g., .../output)
    env_file: str,
    # Parameters needed for path generation and saving
    my_benchmark: BenchmarkGenerationWrapper,
    answering_parameters: Dict[str, Any],
    data_folder: Path,
    # --- Existing parameters ---
    answering_model_name: str = None,
    embedding_model_name: str = None, # Keep for reference
    max_new_tokens: int = 1024, # Keep for reference, might be configured via settings
    subset_fraction: float = 1.0,
    random_seed: int = 42,
    sleeping_time: int = 0, # Less relevant with async, but keep for consistency
    community_level: int = 2 # GraphRAG parameter
) -> pd.DataFrame:
    """
    Generates answers for a given set of questions using GraphRAG and saves them individually as JSON.
    Uses the `graphrag.api` pattern with parallel asyncio execution limited by MAX_WORKERS.
    Structure mimics generator_answers_1_prompting.py more closely.

    Args:
        df_questions: DataFrame containing the questions ('q_idx', 'question').
        graph_components_dir: Path to the directory containing graph components (ROOT of GraphRAG output).
        env_file: Path to the .env file.
        my_benchmark: The BenchmarkGenerationWrapper object.
        answering_parameters: Dictionary of answering parameters (used for paths, models).
        data_folder: Path to the main data folder.
        answering_model_name: Name of the LLM to use for final answering (overrides defaults/settings).
        embedding_model_name: Name of the model used for embedding (informational).
        max_new_tokens: Max tokens for the generated answer (informational).
        subset_fraction: Fraction of questions to process.
        random_seed: Random seed for subset selection.
        sleeping_time: Delay between API calls (optional, less relevant with async).
        community_level: The community level to use in GraphRAG queries.

    Returns:
        DataFrame with generated answers ('q_idx', 'llm_answer', 'retrieved_context').
    """
    print(f"Starting GraphRAG answer generation (using graphrag.api, parallel asyncio)... ")
    start_time = time.time()

    # Load environment variables
    load_dotenv(dotenv_path=env_file)

    # Load MAX_WORKERS from settings
    config = SettingsWrapper(_env_file = env_file)
    max_workers = config.MAX_WORKERS
    print(f"  Using MAX_WORKERS = {max_workers}")

    # Determine model names
    final_answering_model = answering_model_name or answering_parameters.get('model_name', DEFAULT_ANSWERING_MODEL)
    print(f"  Target Answering LLM (Informational for GraphRAG): {final_answering_model}")

    # --- Load GraphRAG Config and Data ONCE ---
    try:
        print(f"  Loading GraphRAG config and data from: {graph_components_dir}")
        root_dir = Path(graph_components_dir).resolve()

        # Check root dir exists
        if not root_dir.is_dir():
             raise FileNotFoundError(f"GraphRAG index root directory not found: {root_dir}")

        # 1. Load Config (assuming settings.yaml is in root_dir)
        graphrag_config = load_config(root_dir)
        print(f"  GraphRAG config loaded successfully.")
        # Note: We might want to override parts of the config here if needed, e.g., model name,
        # but the API might handle this differently or expect it in the saved config.

        # 2. Load required dataframes (entities, communities, etc.)
        output_data_dir = root_dir / "output"
        entities_path = output_data_dir / "entities.parquet"
        communities_path = output_data_dir / "communities.parquet"
        text_units_path = output_data_dir / "text_units.parquet"
        relationships_path = output_data_dir / "relationships.parquet"
        community_reports_path = output_data_dir / "community_reports.parquet"
        covariates_path = output_data_dir / "covariates.parquet" # Optional?

        # Check existence of mandatory files
        required_files = {
            "Entities": entities_path,
            "Communities": communities_path,
            "Text Units": text_units_path,
            "Relationships": relationships_path,
            "Community Reports": community_reports_path
        }
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(f"{name} ({path})")

        if missing_files:
            raise FileNotFoundError(f"Required GraphRAG files not found in {output_data_dir}:\n - " + "\n - ".join(missing_files))

        print("  Loading entities.parquet...")
        entities = pd.read_parquet(entities_path)
        print(f"    Loaded {len(entities)} entities.")
        print("  Loading communities.parquet...")
        communities = pd.read_parquet(communities_path)
        print(f"    Loaded {len(communities)} communities.")
        print("  Loading text_units.parquet...")
        text_units = pd.read_parquet(text_units_path)
        print(f"    Loaded {len(text_units)} text units.")
        print("  Loading relationships.parquet...")
        relationships = pd.read_parquet(relationships_path)
        print(f"    Loaded {len(relationships)} relationships.")
        print("  Loading community_reports.parquet...")
        community_reports = pd.read_parquet(community_reports_path)
        print(f"    Loaded {len(community_reports)} community reports.")

        # Load covariates if it exists
        if covariates_path.exists():
            print("  Loading covariates.parquet...")
            covariates = pd.read_parquet(covariates_path)
            print(f"    Loaded {len(covariates)} covariates.")
        else:
            print("  covariates.parquet not found, proceeding without it.")
            covariates = None # Pass None if file doesn't exist

    except FileNotFoundError as e:
         print(f"ERROR: Required GraphRAG file/directory not found.")
         print(f"       Root Path Checked: {root_dir if 'root_dir' in locals() else 'Not determined'}")
         print(f"       Details: {e}")
         return pd.DataFrame(columns=['q_idx', 'llm_answer', 'retrieved_context'])
    except ImportError as e:
         print(f"ERROR: Missing core GraphRAG components (graphrag.api, graphrag.config).")
         print(f"       Please ensure 'graphrag' is installed correctly.")
         print(f"       Original error: {e}")
         return pd.DataFrame(columns=['q_idx', 'llm_answer', 'retrieved_context'])
    except Exception as e:
        print(f"ERROR: Failed to load GraphRAG config or data: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=['q_idx', 'llm_answer', 'retrieved_context'])

    # --- Prepare Questions DataFrame ---
    # Ensure df_questions has q_idx as a column, not just index
    local_df_questions = df_questions.copy()
    q_idx_in_index = local_df_questions.index.name == 'q_idx'
    q_idx_in_cols = 'q_idx' in local_df_questions.columns

    if 'q_idx' not in local_df_questions.columns:
        if local_df_questions.index.name == 'q_idx':
            local_df_questions = local_df_questions.reset_index()
            print("  Reset index 'q_idx' to column.")
        else:
            raise ValueError("Input df_questions must have a 'q_idx' column or have 'q_idx' as the index name.")
    elif q_idx_in_index: # It's in columns AND was the index
        local_df_questions = local_df_questions.reset_index()
        print("  'q_idx' was both index and column. Resetting index.")
        # Now we might have two 'q_idx' columns

    # Check again if 'q_idx' column exists after potential reset_index
    if 'q_idx' not in local_df_questions.columns:
        raise ValueError("Failed to ensure 'q_idx' is a column after processing index.")

    # --- Deduplicate 'q_idx' column if necessary ---
    q_idx_cols = local_df_questions.columns[local_df_questions.columns == 'q_idx']
    if len(q_idx_cols) > 1:
        print(f"  Warning: Found {len(q_idx_cols)} columns named 'q_idx'. Keeping the first one.")
        # Use pandas built-in method to remove duplicate columns, keeping the first occurrence
        local_df_questions = local_df_questions.loc[:, ~local_df_questions.columns.duplicated(keep='first')]
        print(f"    Columns after deduplication: {local_df_questions.columns.tolist()}")

    # --- Validate the single 'q_idx' column ---
    if 'q_idx' not in local_df_questions.columns:
        raise ValueError("Critical Error: 'q_idx' column lost during deduplication.")
    if isinstance(local_df_questions['q_idx'], pd.DataFrame):
        raise ValueError(f"Critical Error: 'q_idx' is still a DataFrame after deduplication. Columns: {local_df_questions.columns}")

    # *** FIXED: Process 'q_idx' column safely ***
    try:
        # Now this should reliably select a Series
        q_idx_series = local_df_questions['q_idx']

        # Ensure integer type
        if not pd.api.types.is_integer_dtype(q_idx_series):
            print("  Converting 'q_idx' column to integer type.")
            local_df_questions['q_idx'] = q_idx_series.astype(int)
            q_idx_series = local_df_questions['q_idx'] # Re-assign Series after type change

        # Check for uniqueness using the Series attribute
        if not q_idx_series.is_unique:
            print("  Warning: 'q_idx' column contains duplicates. Keeping first occurrence.")
            # Important: drop_duplicates returns a new DataFrame
            local_df_questions = local_df_questions.drop_duplicates(subset=['q_idx'], keep='first')
            print(f"    Dropped duplicates, {len(local_df_questions)} unique questions remaining in local copy.")

    except AttributeError as e_attr:
        print(f"ERROR: An AttributeError occurred while processing 'q_idx'. This likely means it's not a Series.")
        print(f"       Type of local_df_questions['q_idx']: {type(local_df_questions['q_idx'])}")
        print(f"       Original Error: {e_attr}")
        raise ValueError("Failed to prepare 'q_idx' column (AttributeError).") from e_attr
    except Exception as e:
        print(f"ERROR: Could not process 'q_idx' column (convert to int or check uniqueness). Error: {e}")
        traceback.print_exc()
        raise ValueError("Failed to prepare 'q_idx' column in input questions.") from e


    # --- Handle Subset ---
    if subset_fraction < 1.0:
        print(f"  Answering a random subset of {subset_fraction*100:.1f}% of questions.")
        # Sample from the cleaned DataFrame
        df_questions_subset = local_df_questions.sample(frac=subset_fraction, random_state=random_seed)
        print(f"  Subset size: {len(df_questions_subset)} questions.")
    else:
        df_questions_subset = local_df_questions

    # --- Check Cache and Prepare API Calls ---
    processed_rows = [] # List to store results (as dicts) from cache or worker
    tasks_to_run: List[Tuple[int, str]] = []   # List of tuples (q_idx, question) for worker
    total_questions_in_subset = len(df_questions_subset)
    print(f"  Checking cache for {total_questions_in_subset} questions...")

    # Pre-fetch benchmark parameters for path generation
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    for i, row in df_questions_subset.iterrows():
        # q_idx should now be a scalar int because we cleaned the DataFrame above
        q_idx = int(row['q_idx']) # Ensure int
        question = str(row['question']) # Ensure str

        output_filepath = answer_filepath_func(
            q=q_idx, # Use the unique question identifier
            nb_chapters=nb_chapters, nb_tokens=nb_tokens, data_folder=data_folder,
            prompt_parameters=prompt_parameters, model_parameters=model_parameters,
            book_parameters=book_parameters, answering_parameters=answering_parameters
        )

        if output_filepath.is_file():
            # print(f"  Answer file already exists: {output_filepath}")
            try:
                answer = import_list(output_filepath)
                retrieved_context = "Context not retrieved (answer loaded from cache)"
                # Append cached result directly
                processed_row_data = {
                    'q_idx': q_idx,
                    'question': question,
                    'llm_answer': str(answer) if answer is not None else "Processing Skipped/Failed",
                    'retrieved_context': str(retrieved_context)
                }
                processed_rows.append(processed_row_data)
            except Exception as e:
                print(f"  WARNING: Error reading cached answer file {output_filepath}: {e}")
                print(f"           Skipping generation for Q_ID {q_idx} due to cache read error.")
                # Append error entry instead of adding to tasks_to_run
                processed_row_data = {
                    'q_idx': q_idx,
                    'question': question,
                    'llm_answer': "Error reading cached answer",
                    'retrieved_context': "Context unavailable (cache read error)"
                }
                processed_rows.append(processed_row_data)
        else:
            # Not cached, add to list for API call
            tasks_to_run.append((q_idx, question))

    num_queries = len(tasks_to_run)
    print(f"  {num_queries} questions require generation.")

    generated_results = []
    if num_queries > 0:
        # --- ASYNC EXECUTION SETUP ---
        async def main_async(tasks_to_run_async: List[Tuple[int, str]]):
            semaphore = asyncio.Semaphore(max_workers)
            tasks = []
            for q_idx_async, question_async in tasks_to_run_async:
                # Pass all necessary components to the worker task creator
                task = process_single_question_graphrag(
                    q_idx=q_idx_async,
                    question=question_async,
                    graphrag_config=graphrag_config,
                    entities=entities,
                    communities=communities,
                    community_reports=community_reports,
                    text_units=text_units,
                    relationships=relationships,
                    covariates=covariates,
                    nb_chapters=nb_chapters,
                    nb_tokens=nb_tokens,
                    data_folder=data_folder,
                    prompt_parameters=prompt_parameters,
                    model_parameters=model_parameters,
                    book_parameters=book_parameters,
                    answering_parameters=answering_parameters,
                    community_level=community_level,
                    semaphore=semaphore
                )
                tasks.append(task)

            print(f"  Running {len(tasks)} GraphRAG queries concurrently (max {max_workers} workers)... ")
            results = await asyncio.gather(*tasks)
            print(f"  Finished running {len(tasks)} concurrent queries.")
            return results

        # --- Run the async main function ---
        generated_results = asyncio.run(main_async(tasks_to_run))

    # Combine cached results and newly generated results
    processed_rows.extend(generated_results) # generated_results is already a list of dicts

    # Create results DataFrame for the processed subset
    if processed_rows:
        df_results = pd.DataFrame(processed_rows)
    else:
        # Handle case where subset was 0 or all were cached
        df_results = pd.DataFrame(columns=['q_idx', 'llm_answer', 'retrieved_context'])


    # --- Final Merge & Return ---
    # (Merge logic remains largely the same, ensuring 'q_idx' is handled correctly)
    print("  Merging generated/cached answers back with original questions...")
    # Make a copy of the original, *uncleaned* df_questions
    df_questions_final = df_questions.copy()

    # --- Clean df_questions_final before merge (ensure q_idx column) ---
    if 'q_idx' not in df_questions_final.columns:
        if df_questions_final.index.name == 'q_idx':
            df_questions_final = df_questions_final.reset_index()
            print("    Reset index 'q_idx' to column in df_questions_final.")
        else:
            # This case should have been caught earlier, but double-check
            raise ValueError("Original df_questions must have a 'q_idx' column or index name for merging.")

    # Ensure q_idx is int in the original df for merging
    try:
        if 'q_idx' in df_questions_final.columns:
            df_questions_final['q_idx'] = df_questions_final['q_idx'].astype(int)
        else:
            raise ValueError("Critical error: 'q_idx' column missing in df_questions_final before merge.")
    except Exception as e:
        raise ValueError(f"Could not convert 'q_idx' to int in df_questions_final before merge: {e}") from e

    # Drop potential duplicate 'q_idx' columns if they arose, keeping the first
    df_questions_final = df_questions_final.loc[:, ~df_questions_final.columns.duplicated(keep='first')]

    # Check uniqueness *after* ensuring it's an int column
    # No need to drop here, merge 'left' handles it, but warn if needed.
    if not df_questions_final['q_idx'].is_unique:
        print("  Warning: 'q_idx' in original df_questions is not unique before merge. Merge might be ambiguous.")

    # --- Clean df_results before merge (ensure q_idx column and type) ---
    if not df_results.empty:
        if 'q_idx' not in df_results.columns:
            raise ValueError("Internal Error: df_results missing 'q_idx' column after processing.")

        # Ensure q_idx is integer type (should be already from worker)
        try:
            df_results['q_idx'] = df_results['q_idx'].astype(int)
        except Exception as e:
            raise ValueError(f"Could not convert 'q_idx' to int in df_results before merge: {e}") from e

        # Drop potential duplicate 'q_idx' columns from results (shouldn't happen if pre-cleaning worked)
        if not df_results['q_idx'].is_unique:
            print("  Warning: 'q_idx' in df_results is not unique before merge. Dropping duplicates based on 'q_idx'.")
            df_results = df_results.drop_duplicates(subset=['q_idx'], keep='first')
    else: # If df_results is empty (e.g., all subset cached), prepare for merge
        df_results = pd.DataFrame(columns=['q_idx', 'llm_answer', 'retrieved_context']) # Create empty df with needed cols

    # Select only the new columns generated by this function from df_results
    merge_cols = ['q_idx']
    placeholder_row = {}
    if 'llm_answer' in df_results.columns:
        merge_cols.append('llm_answer')
        # Placeholder depends on whether subsetting happened or not
        placeholder_value_answer = "Not Processed (Subset)" if subset_fraction < 1.0 else "Processing Skipped/Failed"
        placeholder_row['llm_answer'] = placeholder_value_answer
    if 'retrieved_context' in df_results.columns:
        merge_cols.append('retrieved_context')
        placeholder_value_context = "Not Processed (Subset)" if subset_fraction < 1.0 else "Context Unavailable"
        placeholder_row['retrieved_context'] = placeholder_value_context

    # Ensure merge_cols are unique
    merge_cols = list(dict.fromkeys(merge_cols))
    df_results_to_merge = df_results[merge_cols]

    # Perform the merge, keeping all original questions
    df_final = pd.merge(df_questions_final, df_results_to_merge, on='q_idx', how='left') # indicator=True is useful for debugging

    # Explicitly fill NaN values in the new columns for rows not processed
    for col, fill_val in placeholder_row.items():
        if col in df_final.columns: # Check if column exists after merge
            # Fillna is generally safer than .loc for this
            df_final[col] = df_final[col].fillna(fill_val)
        else: # Should not happen if merge_cols logic is correct
            print(f"  Warning: Column '{col}' expected but not found after merge. Adding with fill value.")
            df_final[col] = fill_val


    end_time = time.time()
    print(f"GraphRAG answer generation finished in {end_time - start_time:.2f} seconds.")

    return df_final 