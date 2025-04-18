import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
import json # For storing complex attributes like embeddings
import traceback
import pandas as pd # Added for DataFrame and Parquet
from pathlib import Path # Added for path handling
import lancedb # Import lancedb
# Added for community detection (optional, use networkx's built-in or install python-louvain)
# import community as community_louvain # Example if using python-louvain

# --- Helper Functions (Placeholders/Examples) ---

def run_community_detection(entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Placeholder for community detection. Returns communities_df and number of communities.
       Simple implementation: Assigns all nodes to community 0.
    """
    print("  Running simple community detection (All nodes -> community 0)...")
    n_communities = 0
    communities_data = []
    if not entities_df.empty:
        # Use 'id' column from entities_df for entity IDs
        for entity_id in entities_df['id']:
            communities_data.append({"entity_id": entity_id, "community_id": 0})
        n_communities = 1 # We created one community
        print(f"    Assigned {len(entities_df)} entities to community 0.")
        communities_df = pd.DataFrame(communities_data)
    else:
        print("    No entities found, skipping community assignment.")
        communities_df = pd.DataFrame(columns=["entity_id", "community_id"])

    print(f"    Detected {n_communities} communities.")
    return communities_df, n_communities

def generate_community_reports(communities_df: pd.DataFrame, entities_df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for generating community reports.
       Simple implementation: Creates a dummy report for community 0 if it exists.
    """
    print("  Generating simple community reports...")
    report_data = []
    if not communities_df.empty and 0 in communities_df['community_id'].unique():
        community_0_entities = communities_df[communities_df['community_id'] == 0]['entity_id'].tolist()
        num_entities = len(community_0_entities)
        # Example report data - could include top entities, summary stats, etc.
        report_data.append({
            "community_id": 0,
            "report_data": json.dumps({ # Store complex report data as JSON string
                "name": "Community 0 (Default)", 
                "size": num_entities,
                "summary": f"Default community containing {num_entities} entities.",
                "top_entities": community_0_entities[:5] # Example: first 5 entities
            }) 
        })
        print(f"    Generated report for community 0.")
        community_reports_df = pd.DataFrame(report_data)
    else:
        print("    No communities found, skipping report generation.")
        community_reports_df = pd.DataFrame(columns=["community_id", "report_data"])
        
    return community_reports_df

def create_final_nodes(entities_df: pd.DataFrame, communities_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Creates the final nodes table. 
       Simple implementation: Uses initial entities and merges community info.
    """
    print("  Creating final nodes table...")
    # For now, just use entities_df. Add entity resolution logic if needed.
    final_nodes_df = entities_df.copy()
    if communities_df is not None and not communities_df.empty:
        print("    Merging community information into nodes...")
        # Ensure entity_id is the index or a column for merging
        if 'entity_id' not in communities_df.columns:
            communities_df = communities_df.reset_index() # Make entity_id a column if it was index
        if 'id' not in final_nodes_df.columns:
             final_nodes_df = final_nodes_df.reset_index() # Ensure 'id' is a column
             
        # Rename entity_id to id if necessary in communities_df before merge
        if 'entity_id' in communities_df.columns and 'id' not in communities_df.columns:
             communities_df = communities_df.rename(columns={'entity_id': 'id'})
             
        # Ensure the merge key 'id' exists in both DataFrames
        if 'id' in final_nodes_df.columns and 'id' in communities_df.columns:
            final_nodes_df = pd.merge(final_nodes_df, communities_df[['id', 'community_id']], on='id', how='left')
            # Fill NA and ensure int type for community_id
            final_nodes_df['community_id'] = final_nodes_df['community_id'].fillna(-1).astype(int) 
            print(f"    Successfully merged community IDs.")
        else:
             print("    Warning: Could not merge community IDs. Key 'id' missing in entities or communities DataFrame.")
             final_nodes_df['community_id'] = -1 # Assign default if merge fails
    else:
        print("    No community information to merge.")
        final_nodes_df['community_id'] = -1 # No communities detected or provided

    print(f"    Generated final nodes table with {len(final_nodes_df)} nodes.")
    return final_nodes_df

def create_final_relationships(relationships_df: pd.DataFrame) -> pd.DataFrame:
    """Creates the final relationships table.
       Simple implementation: Returns the initial relationships.
    """
    print("  Creating final relationships table (using initial relationships)...")
    # May involve filtering or augmentation later
    final_relationships_df = relationships_df.copy()
    return final_relationships_df

def calculate_covariates(final_nodes_df: pd.DataFrame, final_relationships_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates node covariates.
       Simple implementation: Calculates degree using NetworkX.
    """
    print("  Calculating covariates (degree)...")
    
    if final_nodes_df.empty or final_relationships_df.empty:
        print("    Skipping covariate calculation: No nodes or relationships.")
        # Return DataFrame with node IDs and default degree 0
        node_ids = final_nodes_df['id'].tolist() if not final_nodes_df.empty else []
        return pd.DataFrame({'node_id': node_ids, 'degree': 0, 'in_degree': 0, 'out_degree': 0}) 

    try:
        # Create a directed graph from relationships to calculate in/out degree
        G = nx.DiGraph()
        # Add nodes first to ensure all nodes from final_nodes_df are included
        for node_id in final_nodes_df['id']:
            G.add_node(node_id)
        # Add edges from relationships
        for _, row in final_relationships_df.iterrows():
            # Ensure source/target exist as nodes before adding edge
            if row['source'] in G and row['target'] in G:
                 G.add_edge(row['source'], row['target'])
            else:
                 # This case might happen if relationships refer to nodes not in final_nodes_df
                 # (e.g., if entity resolution removed some nodes)
                 # Decide how to handle: skip edge, add missing nodes?
                 print(f"    Warning: Skipping edge ({row['source']} -> {row['target']}) due to missing node in graph used for covariates.")

        # Calculate degrees
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        degrees = dict(G.degree())

        # Create covariates DataFrame
        node_ids = final_nodes_df['id'].tolist() # Use IDs from the nodes DataFrame
        covariates_data = []
        for node_id in node_ids:
            covariates_data.append({
                'node_id': node_id,
                'degree': degrees.get(node_id, 0),
                'in_degree': in_degrees.get(node_id, 0),
                'out_degree': out_degrees.get(node_id, 0)
            })
            
        covariates_df = pd.DataFrame(covariates_data)
        print(f"    Calculated degree covariates for {len(covariates_df)} nodes.")
        
    except Exception as e:
        print(f"    Error calculating covariates: {e}")
        print(traceback.format_exc())
        # Fallback: return DataFrame with default degree 0
        node_ids = final_nodes_df['id'].tolist()
        covariates_df = pd.DataFrame({'node_id': node_ids, 'degree': 0, 'in_degree': 0, 'out_degree': 0})
        
    return covariates_df

def setup_lancedb(output_dir: Path, text_units_df: pd.DataFrame, embeddings: List[List[float]]):
    """Sets up LanceDB and adds text units with embeddings."""
    print("  Setting up LanceDB...")
    lancedb_uri = output_dir / "lancedb"
    db = lancedb.connect(lancedb_uri)
    print(f"    LanceDB connected at: {lancedb_uri}")

    # Prepare data for LanceDB: Need 'vector' column
    if len(text_units_df) != len(embeddings):
        print(f"    Warning: Mismatch between text units ({len(text_units_df)}) and embeddings ({len(embeddings)}). Skipping LanceDB population.")
        return
        
    # Create a DataFrame copy to avoid modifying the original one
    lancedb_data_df = text_units_df.copy()
    # Ensure embeddings are lists, not JSON strings, for LanceDB
    lancedb_data_df['vector'] = embeddings 
    # Remove the json embedding string if it exists to avoid duplication/confusion
    if 'embedding' in lancedb_data_df.columns:
       lancedb_data_df = lancedb_data_df.drop(columns=['embedding'])

    # Convert DataFrame to list of dicts
    lancedb_data = lancedb_data_df.to_dict('records')

    # Create LanceDB table (adjust schema if needed)
    table_name = "text_units"
    try:
        # Check if table exists, delete if it does for overwrite behavior
        if table_name in db.table_names():
            print(f"    Table '{table_name}' already exists. Dropping and recreating.")
            db.drop_table(table_name)
        
        print(f"    Creating LanceDB table '{table_name}'...")
        # Infer schema or define explicitly for more control
        # table = db.create_table(table_name, schema=...) 
        table = db.create_table(table_name, data=lancedb_data)
        print(f"    Successfully created and populated table '{table_name}' with {len(table)} records.")
    except Exception as e:
        print(f"    Error setting up LanceDB table '{table_name}': {e}")
        print(traceback.format_exc())

# --- Main Builder Function ---

def build_knowledge_graph(
    texts: List[str],
    embeddings: List[List[float]], # Keep original embeddings separate
    extracted_data: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    output_dir: Path,
    run_context: Dict[str, Any] = None # Optional context for context.json
):
    """
    Builds GraphRAG components (final nodes, relationships, communities, covariates, text units)
    and saves them to Parquet files and a LanceDB database.

    Args:
        texts: List of text chunks/units.
        embeddings: List of embeddings corresponding to the texts (used for LanceDB).
        extracted_data: List of tuples from the extractor.
        output_dir: The directory to save the output files and LanceDB data.
        run_context: Optional dictionary with metadata about the run.
    """
    print(f"Building GraphRAG components and saving to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Process Initial Extractions --- 
    print("Step 1: Processing initial extractions...")
    all_entities = {}
    all_relationships = []
    all_text_units = [] # Will store ID and text, embedding goes to LanceDB

    if len(texts) != len(embeddings) or len(texts) != len(extracted_data):
        raise ValueError("Input lists (texts, embeddings, extracted_data) must have the same length.")

    for i, (text, embedding, (entities, relationships)) in enumerate(zip(texts, embeddings, extracted_data)):
        text_unit_id = f"text_unit_{i}"
        # Store only ID and text here. Embedding handled separately for LanceDB.
        all_text_units.append({"id": text_unit_id, "text": text}) 

        for entity in entities:
            entity_id = entity.get("id")
            if not entity_id: continue
            if entity_id not in all_entities:
                 all_entities[entity_id] = {"id": entity_id, "type": entity.get("type"), "name": entity.get("name", entity_id), "mentions": []}
            all_entities[entity_id]["mentions"].append(text_unit_id)

        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            if not source_id or not target_id: continue
            # Basic check if source/target were added (can be made more robust)
            if source_id in all_entities and target_id in all_entities:
                 all_relationships.append({
                     "source": source_id, "target": target_id, "type": rel.get("type"),
                     "description": rel.get("description"), "text_unit_id": text_unit_id
                 })

    # Convert to initial DataFrames
    text_units_df = pd.DataFrame(all_text_units)
    entities_list = list(all_entities.values())
    for entity in entities_list:
        entity['mentions'] = json.dumps(entity.get('mentions', []))
    initial_entities_df = pd.DataFrame(entities_list)
    initial_relationships_df = pd.DataFrame(all_relationships)
    print(f"  Processed {len(text_units_df)} text units, {len(initial_entities_df)} initial entities, {len(initial_relationships_df)} initial relationships.")

    # --- 2. Community Detection & Reporting --- 
    print("\nStep 2: Running community detection...")
    # Pass initial entities/relationships DFs to community detection
    communities_df, n_communities = run_community_detection(initial_entities_df, initial_relationships_df)
    community_reports_df = generate_community_reports(communities_df, initial_entities_df)

    # --- 3. Create Final Graph Components --- 
    print("\nStep 3: Creating final graph components...")
    # Pass initial entities and communities to final node creation
    final_nodes_df = create_final_nodes(initial_entities_df, communities_df)
    final_relationships_df = create_final_relationships(initial_relationships_df)
    # Pass final nodes/relationships to covariate calculation
    covariates_df = calculate_covariates(final_nodes_df, final_relationships_df)
    
    # Add final entities df (maybe same as final_nodes or derived differently? Using initial for now)
    # TODO: Clarify if create_final_entities is needed or if final_nodes_df suffices
    final_entities_df = initial_entities_df.copy() 

    # --- 4. Setup LanceDB --- 
    print("\nStep 4: Setting up LanceDB...")
    try:
        # Pass the original embeddings list here
        setup_lancedb(output_dir, text_units_df, embeddings)
    except ImportError:
        print("  Error: lancedb library not found. Please install it (`pip install lancedb`). Skipping LanceDB setup.")
    except Exception as ldb_e:
        print(f"  Error during LanceDB setup: {ldb_e}")
        print(traceback.format_exc())

    # --- 5. Calculate Final Stats --- 
    print("\nStep 5: Calculating final statistics...")
    stats = {
        "num_text_units": len(text_units_df),
        "num_final_nodes": len(final_nodes_df),
        "num_final_entities": len(final_entities_df), # Using placeholder
        "num_final_relationships": len(final_relationships_df),
        "num_communities": n_communities,
        "num_covariates": len(covariates_df.columns) -1 if not covariates_df.empty else 0, # -1 for node_id
    }
    print(f"  Stats: {stats}")

    # --- 6. Save Final Outputs --- 
    print("\nStep 6: Saving final output files...")
    # Define final output files with the new prefix
    output_files = {
        "create_final_text_units.parquet": text_units_df,
        "create_final_entities.parquet": final_entities_df, # Using placeholder
        "create_final_nodes.parquet": final_nodes_df,
        "create_final_relationships.parquet": final_relationships_df,
        "create_final_communities.parquet": communities_df,
        "create_final_community_reports.parquet": community_reports_df,
        "create_final_covariates.parquet": covariates_df,
    }

    for filename, df in output_files.items():
        filepath = output_dir / filename
        try:
            if df is not None and not df.empty:
                df.to_parquet(filepath, index=False)
                print(f"  Successfully saved {filepath}")
            else:
                print(f"  Skipping empty or None DataFrame for {filename}")
        except Exception as e:
            print(f"  Error saving {filename} to Parquet: {e}")
            print(traceback.format_exc())

    # Save stats and context JSON (keeping original names for these)
    stats_path = output_dir / "stats.json"
    context_path = output_dir / "context.json"
    try:
        with open(stats_path, 'w') as f: json.dump(stats, f, indent=4)
        print(f"  Successfully saved {stats_path}")
    except Exception as e: print(f"  Error saving stats.json: {e}")
    try:
        context_to_save = run_context if run_context is not None else {}
        context_to_save["output_directory"] = str(output_dir)
        with open(context_path, 'w') as f: json.dump(context_to_save, f, indent=4, default=str)
        print(f"  Successfully saved {context_path}")
    except Exception as e: print(f"  Error saving context.json: {e}")

    print("GraphRAG component generation complete.")
    return stats

# --- Example Usage (Updated) ---
if __name__ == '__main__':
    print("Running builder example...")
    # Sample Data (same as before)
    sample_texts = [
        "Chunk 1 text about Alice and Library.",
        "Chunk 2 text about Bob and Alice at the Park."
    ]
    sample_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    sample_extracted_data = [
        ([{"id": "Alice", "type": "Person", "name": "Alice"}, {"id": "Library", "type": "Location", "name": "library"}],
         [{"source": "Alice", "target": "Library", "type": "LOCATED_AT"}]),
        ([{"id": "Bob", "type": "Person", "name": "Bob"}, {"id": "alice", "type": "Person", "name": "Alice"}, {"id": "Park", "type": "Location", "name": "park"}],
         [{"source": "Bob", "target": "alice", "type": "MET"}, {"source": "Bob", "target": "Park", "type": "LOCATED_AT"}])
    ]
    example_output_dir = Path("./epbench_output_example_graphrag")
    example_context = {"run_id": "graphrag_example_001"}

    print(f"Output will be saved to: {example_output_dir.resolve()}")
    
    try:
        print("\nCalling build_knowledge_graph (GraphRAG version)...")
        run_stats = build_knowledge_graph(
            texts=sample_texts,
            embeddings=sample_embeddings,
            extracted_data=sample_extracted_data,
            output_dir=example_output_dir,
            run_context=example_context
        )
        print("\nbuild_knowledge_graph finished.")
        if run_stats:
            print("Run Statistics:")
            for key, value in run_stats.items(): print(f"  {key}: {value}")

        # Verify Output Files and Directory
        print("\nChecking for output files and directory:")
        expected_files = [
            "create_final_text_units.parquet", "create_final_entities.parquet", 
            "create_final_nodes.parquet", "create_final_relationships.parquet",
            "create_final_communities.parquet", "create_final_community_reports.parquet",
            "create_final_covariates.parquet", "stats.json", "context.json"
        ]
        # Check for lancedb directory separately
        lancedb_dir = example_output_dir / "lancedb"
        if lancedb_dir.is_dir():
             print(f"  [FOUND DIR] lancedb")
        else:
             # Check if LanceDB was skipped due to import error
             # This requires checking logs or adding a flag, simplified check for now
             print(f"  [MISSING DIR] lancedb (May be missing due to import error)") 
             
        for fname in expected_files:
            fpath = example_output_dir / fname
            if fpath.exists():
                print(f"  [FOUND FILE] {fname}")
            else:
                # Communities/reports/covariates might be empty/skipped by placeholders
                if any(sub in fname for sub in ["communities", "reports", "covariates"]):
                     print(f"  [MISSING/EMPTY] {fname} (Potentially due to placeholder implementation)")
                else:
                     print(f"  [MISSING FILE] {fname} - Check implementation.")

    except ImportError as ie:
         print(f"\nImportError: {ie}. Please ensure required libraries (pandas, pyarrow, lancedb) are installed.")
    except ValueError as ve:
        print(f"\nError during graph building: {ve}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example run: {e}")
        print(traceback.format_exc()) 