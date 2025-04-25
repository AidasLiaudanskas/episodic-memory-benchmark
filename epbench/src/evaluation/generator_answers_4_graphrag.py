#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate answers using a GraphRAG approach."""

import os
import time
import traceback
import asyncio # Added for running async API calls
from pathlib import Path
from typing import Dict, Any, List
import logging

# ADDED: Configure logging to suppress INFO messages from graphrag
logging.getLogger("graphrag").setLevel(logging.WARNING)

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

# --- Constants ---
DEFAULT_ANSWERING_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small" # Keep for reference

# --- Main Function ---

def generate_answers_graphrag(
    df_questions: pd.DataFrame,
    graph_components_dir: Path, # This is the specific path to the graphrag index (e.g., .../output)
    env_file: str,
    # ADDED: Parameters needed for path generation and saving
    my_benchmark: BenchmarkGenerationWrapper,
    answering_parameters: Dict[str, Any],
    data_folder: Path,
    # --- Existing parameters ---
    answering_model_name: str = None,
    embedding_model_name: str = None, # Keep for reference
    max_new_tokens: int = 1024, # Keep for reference, might be configured via settings
    subset_fraction: float = 1.0,
    random_seed: int = 42,
    sleeping_time: int = 0,
    community_level: int = 2 # GraphRAG parameter - assuming QueryEngine uses it
) -> pd.DataFrame:
    """
    Generates answers for a given set of questions using GraphRAG and saves them individually as JSON.
    Uses the `graphrag.api` pattern.

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
        sleeping_time: Delay between API calls (optional).
        community_level: The community level to use in GraphRAG queries.

    Returns:
        DataFrame with generated answers ('q_idx', 'llm_answer', 'retrieved_context').
    """
    print(f"Starting GraphRAG answer generation (using graphrag.api)...")
    start_time = time.time()

    # Load environment variables
    load_dotenv(dotenv_path=env_file)

    # Determine model names
    final_answering_model = answering_model_name or answering_parameters.get('model_name', DEFAULT_ANSWERING_MODEL)
    print(f"  Target Answering LLM: {final_answering_model}")

    # --- Load GraphRAG Config and Data ---
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

        # 2. Load required dataframes (entities, communities)
        entities_path = root_dir / "output" / "entities.parquet"
        communities_path = root_dir / "output" / "communities.parquet"
        # ADDED: Paths for other required dataframes
        text_units_path = root_dir / "output" / "text_units.parquet"
        relationships_path = root_dir / "output" / "relationships.parquet"
        community_reports_path = root_dir / "output" / "community_reports.parquet"
        covariates_path = root_dir / "output" / "covariates.parquet" # Optional?
        
        # Check existence of mandatory files
        required_files = {
            "Entities": entities_path,
            "Communities": communities_path,
            "Text Units": text_units_path,
            "Relationships": relationships_path,
            "Community Reports": community_reports_path
        }
        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")

        print("  Loading entities.parquet...")
        entities = pd.read_parquet(entities_path)
        print(f"    Loaded {len(entities)} entities.")
        print("  Loading communities.parquet...")
        communities = pd.read_parquet(communities_path)
        print(f"    Loaded {len(communities)} communities.")
        # ADDED: Load other required dataframes
        print("  Loading text_units.parquet...")
        text_units = pd.read_parquet(text_units_path)
        print(f"    Loaded {len(text_units)} text units.")
        print("  Loading relationships.parquet...")
        relationships = pd.read_parquet(relationships_path)
        print(f"    Loaded {len(relationships)} relationships.")
        print("  Loading community_reports.parquet...")
        community_reports = pd.read_parquet(community_reports_path)
        print(f"    Loaded {len(community_reports)} community reports.")

        # ADDED: Load covariates if it exists
        if covariates_path.exists():
            print("  Loading covariates.parquet...")
            covariates = pd.read_parquet(covariates_path)
            print(f"    Loaded {len(covariates)} covariates.")
        else:
            print("  covariates.parquet not found, proceeding without it.")
            covariates = None # Pass None if file doesn't exist
        
        # Global search data (not needed for local search loop)
        # community_reports_path = root_dir / "output" / "community_reports.parquet"
        # community_reports = pd.read_parquet(community_reports_path) if community_reports_path.exists() else None

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

    # Handle subset if needed
    if subset_fraction < 1.0:
        print(f"Answering a random subset of {subset_fraction*100:.1f}% of questions.")
        df_questions_subset = df_questions.sample(frac=subset_fraction, random_state=random_seed)
    else:
        df_questions_subset = df_questions

    # results_list = [] # REMOVED: We will build the final DataFrame differently
    processed_rows = [] # ADDED: List to store processed rows (as dicts or Series)
    total_questions = len(df_questions_subset)
    print(f"Processing {total_questions} questions...")

    # Pre-fetch benchmark parameters
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    # --- Prepare tasks for uncached questions --- 
    tasks_to_run = []
    uncached_indices = [] # Keep track of original index for matching results
    uncached_output_paths = []
    cached_rows = [] # Store rows loaded from cache

    # --- Define async helper for API call (moved outside loop) ---
    async def run_local_search_task(query_text: str):
        # This function now only wraps the API call for gather
        # Error handling will happen when processing gather results
        return await api.local_search(
            config=graphrag_config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates, 
            query=query_text,
            community_level=community_level,
            response_type="Multiple Paragraphs" 
        )

    print("  Checking cache and preparing tasks...")
    # --- First Pass: Check cache and prepare tasks/cached data ---
    for i, row in df_questions_subset.iterrows():
        q_idx = row['q_idx']
        question = row['question']
        
        # Calculate output path using the original index 'i' from the subsetted df
        output_filepath = answer_filepath_func(
            q=i, nb_chapters=nb_chapters, nb_tokens=nb_tokens, data_folder=data_folder,
            prompt_parameters=prompt_parameters, model_parameters=model_parameters,
            book_parameters=book_parameters, answering_parameters=answering_parameters
        )

        if output_filepath.is_file():
            # Load from cache
            try:
                answer = import_list(output_filepath)
                retrieved_context = "Context not retrieved (answer loaded from cache)"
                print(f"  Q{i+1}/{total_questions} (ID: {q_idx}): Loaded from cache.")
                # Store the complete row data
                cached_row_data = row.to_dict()
                cached_row_data['llm_answer'] = str(answer)
                cached_row_data['retrieved_context'] = retrieved_context
                cached_rows.append(cached_row_data)
            except Exception as e:
                print(f"  Q{i+1}/{total_questions} (ID: {q_idx}): Error reading cache {output_filepath}: {e}. Will regenerate.")
                # Prepare task for regeneration if cache read fails
                tasks_to_run.append(run_local_search_task(question))
                uncached_indices.append(i) # Use original index 'i'
                uncached_output_paths.append(output_filepath)
        else:
            # Prepare task for generation
            print(f"  Q{i+1}/{total_questions} (ID: {q_idx}): Needs generation.")
            tasks_to_run.append(run_local_search_task(question))
            uncached_indices.append(i) # Use original index 'i'
            uncached_output_paths.append(output_filepath)

    # --- Run uncached queries concurrently --- 
    newly_processed_rows = []
    if tasks_to_run:
        print(f"  Running {len(tasks_to_run)} queries concurrently using asyncio.gather...")
        start_gather_time = time.time()

        async def run_gather():
            # Use return_exceptions=True to handle errors in individual tasks
            return await asyncio.gather(*tasks_to_run, return_exceptions=True)

        # Run the gather operation
        results = asyncio.run(run_gather())
        
        end_gather_time = time.time()
        print(f"  Finished {len(tasks_to_run)} queries in {end_gather_time - start_gather_time:.2f} seconds.")

        # --- Process results and save --- 
        print("  Processing and saving results...")
        for idx, result in enumerate(results):
            original_index = uncached_indices[idx]
            output_filepath = uncached_output_paths[idx]
            row_data = df_questions_subset.loc[original_index].to_dict() # Get original row data using index
            q_idx = row_data['q_idx'] # Get q_idx for messages

            answer = None
            retrieved_context = None

            if isinstance(result, Exception):
                # Handle exceptions returned by gather
                print(f"  ERROR during GraphRAG query for Q index {original_index} (ID: {q_idx}): {result}")
                answer = f"Error during GraphRAG query: {result}"
                retrieved_context = f"Error during GraphRAG processing: {result}"
            else:
                # Unpack successful result
                try:
                    answer, retrieved_context = result
                    print(f"    Q index {original_index} (ID: {q_idx}) Answer (preview): {str(answer)[:100]}...")
                    # Save the newly generated answer
                    answer_to_save = str(answer) if answer is not None else "Processing Skipped/Failed"
                    try:
                        output_filepath.parent.mkdir(parents=True, exist_ok=True)
                        export_list(answer_to_save, output_filepath)
                        # print(f"    Successfully saved answer to: {output_filepath}")
                    except Exception as e:
                        print(f"    Error saving answer for Q index {original_index} (ID: {q_idx}) to JSON: {e}")
                except Exception as unpack_error:
                     print(f"  ERROR unpacking result for Q index {original_index} (ID: {q_idx}): {unpack_error}")
                     answer = f"Error unpacking result: {unpack_error}"
                     retrieved_context = "Error unpacking result"

            # Append the processed row data
            row_data['llm_answer'] = str(answer) if answer is not None else "Processing Skipped/Failed"
            row_data['retrieved_context'] = str(retrieved_context) if retrieved_context is not None else "Context Unavailable"
            newly_processed_rows.append(row_data)
    else:
        print("  No queries needed to run (all answers found in cache).")

    # Combine cached and newly processed rows
    all_processed_rows = cached_rows + newly_processed_rows

    end_time = time.time()
    print(f"GraphRAG answer generation finished in {end_time - start_time:.2f} seconds.")

    # --- Create final DataFrame --- 
    if not all_processed_rows:
        print("Warning: No rows were processed (cached or generated). Returning empty DataFrame structure.")
        # Return original columns plus the ones we expect to add
        final_columns = list(df_questions.columns) + ['llm_answer', 'retrieved_context']
        df_final = pd.DataFrame(columns=final_columns)
    else:
        df_results = pd.DataFrame(all_processed_rows)

        # If subsetting was used, merge back with original to ensure all questions are present
        if subset_fraction < 1.0:
            merge_cols = list(df_results.columns) # Use all columns from results
            # Ensure original df_questions has a unique index if it wasn't already
            if not df_questions.index.is_unique:
                 df_questions = df_questions.reset_index(drop=True)
            # Merge, keeping all original questions. Use suffixes to handle potential column conflicts (though unlikely now)
            df_final = pd.merge(df_questions, df_results[merge_cols], on='q_idx', how='left', suffixes=('', '_new'))
            # Clean up potential duplicated columns if merge added suffixes unexpectedly (shouldn't happen with on='q_idx')
            # Example: if 'question_new' exists, drop it if 'question' exists.
            cols_to_drop = [col for col in df_final.columns if col.endswith('_new')]
            df_final = df_final.drop(columns=cols_to_drop)

        else:
            df_final = df_results # No merge needed if all questions were processed

    # Ensure required columns exist even if generation failed completely or no rows processed
    if 'llm_answer' not in df_final.columns:
         df_final['llm_answer'] = "Processing Skipped/Failed"
    if 'retrieved_context' not in df_final.columns:
         df_final['retrieved_context'] = "Context Unavailable"

    return df_final 