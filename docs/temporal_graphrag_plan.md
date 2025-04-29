# Plan for Adding Temporal Features to GraphRAG

This document outlines the steps required to integrate temporal awareness into the GraphRAG indexing and retrieval processes.

## 1. Data Model Changes (`graphrag/data_model/relationship.py`)

-   **Goal:** Add fields to the `Relationship` dataclass to store temporal information.
-   **Action:**
    -   Introduce a `timestamp` field. Using `str | None` initially might offer flexibility for different date/time formats extracted by the LLM, potentially standardizing later (e.g., ISO 8601 format `YYYY-MM-DDTHH:MM:SSZ`). Consider adding `valid_from: str | None = None` and `valid_to: str | None = None` as well for time ranges if needed later.
    -   Update the `from_dict` class method to handle the new temporal field(s), including relevant key parameters (e.g., `timestamp_key`).
    -   **Constraint:** Use `str | None` for initial flexibility. Assume ISO 8601 format (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`) as the target format for parsing later.

## 2. Graph Extraction Prompt Update (`graphrag/prompts/index/extract_graph.py`)

-   **Goal:** Modify the LLM prompt (`GRAPH_EXTRACTION_PROMPT`) to instruct the model to extract relevant timestamps or time periods associated with relationships.
-   **Action:**
    -   Locate the `GRAPH_EXTRACTION_PROMPT` variable.
    -   Update the prompt instructions to explicitly ask for a timestamp or date associated with each identified relationship, specifying the desired output format (e.g., "Extract relationships as `('relationship', <source>, <target>, <description>, <timestamp_iso8601>)`. The timestamp should be in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ) or leave empty if unknown.").
    -   Ensure the example output in the prompt reflects the new structure.
    -   **Context:** The LLM's ability to consistently extract and format timestamps accurately might vary. Robust parsing in the next step is crucial.

## 3. Graph Extraction Logic Update (`graphrag/index/operations/extract_graph/graph_extractor.py`)

-   **Goal:** Update the `GraphExtractor._process_results` method to parse and store the extracted temporal information onto the graph edges.
-   **Sub-steps:**
    -   **3.1. Adjust Attribute Parsing:** Modify the line `record_attributes = record.split(tuple_delimiter)` and subsequent checks (e.g., `len(record_attributes) >= 5`) to expect the new timestamp element at the end of the relationship tuple (e.g., index 4 if description is index 3).
    -   **3.2. Extract Raw Timestamp:** Extract the raw timestamp string from `record_attributes` (e.g., `raw_timestamp_str = record_attributes[4] if len(record_attributes) >= 5 else None`).
    -   **3.3. Clean Timestamp:** Apply basic cleaning using `clean_str` to the `raw_timestamp_str`.
    -   **3.4. Add Timestamp to Edge:** When adding a new edge (`graph.add_edge(...)`), include the cleaned timestamp string as a new attribute: `timestamp=cleaned_timestamp_str`.
    -   **3.5. Update Edge Timestamp Logic:** Modify the logic within the `if graph.has_edge(source, target):` block. Decide on a strategy for handling timestamps when merging edge data.
        -   **Initial Strategy:** Overwrite the existing `timestamp` with the new one if the new one is not None/empty. Add `timestamp=cleaned_timestamp_str` (or update `edge_data['timestamp'] = cleaned_timestamp_str`) within this block, similar to how `description` and `weight` are handled.
        -   **Alternative (Consider Later):** Store multiple timestamps in a list or implement more complex logic (e.g., keep earliest/latest).
    -   **Constraint:** For the initial implementation, prioritize simple overwriting for timestamp updates on existing edges. Do not implement complex date validation/parsing *at this stage*; store the cleaned string provided by the LLM.

## 4. Retrieval Logic Update

-   **Goal:** Modify retrieval methods to allow filtering based on temporal constraints.
-   **Context:** This is potentially the most complex part, requiring changes where graph data is queried. The primary target is likely the structured local search, but LLM-based context generation might also need adjustments if it directly uses relationship properties.
-   **Sub-steps:**
    -   **4.1. Identify Target Functions:**
        -   Focus on `graphrag/query/structured_search/local_search/search.py`. Analyze the `LocalSearch.search` method and any helper methods it calls that access/filter graph edges (e.g., methods retrieving neighbors or edges based on criteria).
        -   *Self-Correction/Refinement:* Look for where `networkx` graph methods like `graph.edges(data=True)`, `graph.neighbors`, or `graph[source][target]` are used to fetch relationship data. The filtering needs to happen *after* fetching the data but *before* it's used for ranking or reporting.
    -   **4.2. Modify Function Signatures:** Add `start_date: str | None = None` and `end_date: str | None = None` parameters to the identified function(s) (e.g., `LocalSearch.search`). Ensure these parameters are propagated from the main query input if necessary.
    -   **4.3. Implement Filtering Point:** Locate the place(s) in the target function(s) where a list or iterator of edges/relationships (with their attributes) is available.
    -   **4.4. Add Temporal Filter Logic:**
        -   Check if `start_date` or `end_date` are provided. If not, skip temporal filtering.
        -   If provided, import Python's `datetime` and `logging`.
        -   Iterate through the edges. For each edge:
            -   Retrieve the `timestamp` attribute string from the edge data (e.g., `edge_data.get('timestamp')`).
            -   If the timestamp attribute exists and is not None/empty:
                -   **Attempt Parsing:** Use a `try-except` block to parse the edge's timestamp string into a `datetime` object. Try ISO 8601 formats (e.g., `datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))`). Handle potential `ValueError`. Log errors if parsing fails.
                -   **Attempt Query Date Parsing:** Similarly, parse `start_date` and `end_date` strings into `datetime` objects *once* before the loop. Handle errors.
                -   **Compare:** If edge and query dates are parsed successfully, compare the edge's datetime object against the start/end datetimes. Keep the edge only if it falls within the range (inclusive: `start_datetime <= edge_datetime <= end_datetime`). Handle cases where only start or only end date is provided.
            -   If the edge has no timestamp or it fails parsing, decide on behavior. **Initial Strategy:** Exclude the edge from results if temporal filtering is active and the edge timestamp is missing or unparseable.
        -   Ensure the *result* of this filtering (the subset of edges meeting temporal criteria) is used for subsequent processing (ranking, context building, etc.).
    -   **Constraint:** Use Python's `datetime` for comparison. Initially assume ISO 8601 format for both stored timestamps and query parameters. Handle parsing errors by logging and excluding the problematic edge/skipping the filter if query dates are bad.

## 5. (Optional) Vector Store Integration

-   **Goal:** If using vector stores for relationship or document retrieval, incorporate temporal data.
-   **Sub-steps (High Level):**
    -   **5.1. Indexing:** Identify the data preparation step before indexing (e.g., converting graph elements or documents to records for the vector store). Add the `timestamp` string as a metadata field to these records.
    -   **5.2. Querying:** Locate the vector search calls (e.g., `vector_store.similarity_search`). Consult the specific vector store's documentation (e.g., Chroma, LanceDB) for metadata filtering capabilities (often via a `where` clause or `filter` parameter). Modify the calls to include filters based on the `timestamp` metadata field using the provided `start_date` and `end_date`. Date conversion/comparison might need to adhere to the specific store's requirements.
    -   **Constraint:** This heavily depends on the specific vector store implementation being used. Requires consulting external documentation.

## 6. Configuration (`graphrag/config/defaults.py` and Schema)

-   **Goal:** Allow users to enable/configure temporal features.
-   **Action:**
    -   Add configuration parameters (e.g., `temporal_extraction_enabled: bool = True` (or False by default), `relationship_timestamp_attribute: str = "timestamp"`) possibly under `graph_extraction` or `defaults`.
    -   Update config schema definition if one exists.
    -   *Refinement:* In Step 3 and 4, wrap the temporal-specific logic (parsing timestamp from LLM, adding attribute, filtering during retrieval) in `if config.temporal_extraction_enabled:` checks.
    -   **Constraint:** Start with a simple boolean flag to enable/disable the feature globally.

## 7. Testing

-   **Goal:** Ensure the new temporal features work correctly and don't introduce regressions.
-   **Sub-steps:**
    -   **7.1. Unit Test - Data Model:** Add tests for `Relationship` to ensure the `timestamp` field exists and `from_dict` handles `timestamp_key`.
    -   **7.2. Unit Test - Extraction:** Test `GraphExtractor._process_results` with mock LLM output strings containing relationship tuples *with* timestamps. Verify the `timestamp` attribute is correctly added to `networkx` graph edges, including the overwrite logic for existing edges. Test cases with missing/empty timestamps in the input string.
    -   **7.3. Unit Test - Retrieval Filtering:** Create mock `networkx` graphs with edges having `timestamp` attributes (various valid ISO 8601 strings, some missing, some invalid). Test the temporal filtering logic implemented in Step 4.4 directly by calling the modified retrieval function (or the isolated filter logic) with different `start_date`/`end_date` combinations (including None). Verify correct edges are returned/excluded. Test date parsing robustness.
    -   **7.4. Integration Test:**
        -   Create a small set of input text files (`.txt`) with content mentioning entities and relationships at specific dates (e.g., "Alice met Bob on 2023-01-15.", "Project X started between OrgA and OrgB on 2024-03-10.").
        -   Run the full GraphRAG indexing pipeline (`graphrag --init --root . run ...`) on this dataset, ensuring the modified extraction prompt and logic are used.
        -   Inspect the output graph artifact (e.g., `.graphml` file) to manually verify timestamps are present on edges.
        -   Run queries using the (to-be-modified) query interface or API, specifying `start_date` and `end_date` parameters. Verify that the query results correctly reflect the temporal constraints. For example, a query for "relationships in 2023" should only return the "Alice met Bob" relationship.

## Considerations

-   **Timestamp Granularity & Format:** Standardize on ISO 8601 strings internally. LLM might provide variations; parsing needs to be somewhat flexible but enforce this standard where possible.
-   **Parsing Robustness:** Crucial, especially for LLM output (Step 3.3, Step 4.4). Log errors, decide on fallback (exclude edge, ignore timestamp).
-   **Missing Timestamps:** 
    -   **Current Behavior:** When temporal filtering is *disabled* (either via `temporal_filtering_enabled: false` in config or no `start_date`/`end_date` provided in the query), relationships without a timestamp (missing attribute or empty string) are included in the context building process like any other relationship, subject to standard relevance and token limits.
    -   **Current Behavior:** When temporal filtering is *enabled* and `start_date` or `end_date` *are* provided in the query, relationships that lack a valid, parsable timestamp in their designated attribute (`relationship_timestamp_attribute`) are **excluded** from the pool of relationships considered for the context. They will not appear in the `Relationships` section of the context passed to the LLM.
    -   **Future Consideration:** This exclusion behavior could be made configurable (e.g., a flag `include_timeless_relationships_when_filtering: bool = False`).
-   **Time Zones:** Ignored initially for simplicity. Assume UTC or local time based on input if not specified. Add TZ handling later if required. Requires careful management in parsing and comparison.
-   **Performance:** Filtering adds overhead. For very large graphs, indexing timestamps appropriately (if using databases/vector stores) might be necessary instead of filtering Python-side graph objects every time.

## 8. Testing and epbench Integration Plan

This section outlines how to test the implemented temporal filtering and integrate it with the `epbench` benchmark scripts.

### 8.1. Standalone GraphRAG Integration Testing

-   **Goal:** Verify that temporal filtering works correctly within GraphRAG before involving `epbench`.
-   **Steps:**
    1.  **Create Test Data:** Prepare a small set of input text files (`.txt`) containing sentences with clear dates associated with relationships (e.g., "Alice emailed Bob on 2023-10-26.", "Company A acquired Company B in 2022.", "Project C ran from 2024-01-15 to 2024-06-30.").
    2.  **Install Modified GraphRAG:** Ensure the Python environment uses your modified `graphrag` code (e.g., using `pip install -e .` in the `graphrag` directory).
    3.  **Build Index:** Run the GraphRAG indexing pipeline on your test data:
        ```bash
        graphrag --init --root ./temp_test_index run
        ```
        (Ensure the `settings.yaml` in `./temp_test_index` points to models accessible to you).
    4.  **Verify Extraction:** Inspect the output Parquet files, especially `relationships.parquet`, in `./temp_test_index/output/`. Use `pandas` in a Python script or notebook to check if the `timestamp` column (or the configured attribute name) was populated correctly based on your test data.
        ```python
        import pandas as pd
        rels_df = pd.read_parquet("./temp_test_index/output/relationships.parquet")
        print(rels_df[[\'source\', \'target\', \'timestamp\']])
        ```
    5.  **Modify GraphRAG API (Required for Direct Test):** The current `graphrag.api.local_search` function does not accept `start_date` or `end_date`. You need to modify it:
        *   **File:** `graphrag/graphrag/api.py`
        *   **Action:** Add `start_date: str | None = None` and `end_date: str | None = None` to the `local_search` function signature.
        *   **Action:** Inside `local_search`, pass these new parameters to the `search_engine.search(...)` call within the `kwargs` dictionary, e.g., `search_engine.search(query=query, ..., start_date=start_date, end_date=end_date)`.
    6.  **Test Querying (via Python):** Create a Python script that:
        *   Imports `graphrag.api` and `pandas`.
        *   Loads the configuration: `config = graphrag.config.load_config("./temp_test_index")`
        *   Loads the necessary dataframes (entities, relationships, etc.) from `./temp_test_index/output/`.
        *   Calls the modified `await graphrag.api.local_search(...)` with your test query and different `start_date` and `end_date` values.
        *   Print the `api_response` and `api_context`.
        *   Verify that the response and context reflect the temporal filtering (e.g., only relationships from 2023 are shown when querying with `start_date="2023-01-01"`, `end_date="2023-12-31"`).

### 8.2. Integrate Temporal Filtering into `epbench`

-   **Goal:** Allow `epbench` to leverage the new GraphRAG temporal filtering capabilities, likely by extracting date constraints from questions.
-   **Steps:**
    1.  **Modify `generator_answers_4_graphrag.py`:**
        *   **File:** `episodic-memory-benchmark/epbench/src/evaluation/generator_answers_4_graphrag.py`
        *   **Add Date Extraction Logic:** Inside the `process_single_question_graphrag` async function (or potentially before the loop in `generate_answers_graphrag`), add logic to parse the input `question` string for temporal constraints. Examples:
            *   Use regular expressions (`re` module) to find patterns like "in YEAR", "on YYYY-MM-DD", "between DATE1 and DATE2", "during YEAR", "after DATE", "before DATE".
            *   Define target formats (e.g., always convert extracted dates to `YYYY-MM-DD`).
            *   Store extracted dates in `extracted_start_date` and `extracted_end_date` variables (set to `None` if no pattern matches).
            ```python
            import re
            
            def extract_dates_from_question(question: str) -> tuple[str | None, str | None]:
                start_date, end_date = None, None
                # Example: Match "in YYYY"
                match_year = re.search(r"\b(?:in|during)\s+(\d{4})\b", question, re.IGNORECASE)
                if match_year:
                    year = match_year.group(1)
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"
                    
                # Example: Match "on YYYY-MM-DD"
                match_date = re.search(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", question, re.IGNORECASE)
                if match_date:
                    date = match_date.group(1)
                    start_date = date
                    end_date = date # Treat single date as a range for that day
                    
                # Add more regex for "between ... and ...", "after ...", "before ..."
                # ... (implement more robust parsing) ...
                    
                return start_date, end_date
            
            # Inside process_single_question_graphrag:
            extracted_start_date, extracted_end_date = extract_dates_from_question(question)
            if extracted_start_date or extracted_end_date:
                 with print_lock:
                     print(f"    Extracted dates for Q_ID {q_idx}: start={extracted_start_date}, end={extracted_end_date}")
            ```
        *   **Pass Dates to API:** Modify the call `await api.local_search(...)` to include the `extracted_start_date` and `extracted_end_date` as arguments (assuming the API was modified in Step 8.1.5).
            ```python
            api_response, api_context = await api.local_search(
                # ... existing args ...
                query=question,
                start_date=extracted_start_date, # Pass extracted date
                end_date=extracted_end_date,   # Pass extracted date
                community_level=community_level,
                # ... other args ...
            )
            ```
    2.  **Modify `quickstart.py` (Optional Configuration Override):**
        *   **File:** `episodic-memory-benchmark/epbench/experiments/quickstart.py`
        *   **Add Arguments:** Add argparse arguments for `--temporal_filtering_enabled` (e.g., `action='store_true'/'store_false'`) and potentially `--relationship_timestamp_attribute`.
        *   **Store in Params:** Add these arguments to the `answering_parameters` dictionary.
        *   **Apply Override (in `generator_answers_4_graphrag.py`):** Before calling the API, check if these override flags exist in `answering_parameters`. If they do, modify the loaded `graphrag_config` object *in memory* before passing it to `api.local_search`. Accessing nested config can be tricky; you might need helper functions or direct dictionary manipulation.
            ```python
            # Inside generate_answers_graphrag, after loading config
            override_temporal = answering_parameters.get('temporal_filtering_enabled')
            if override_temporal is not None: 
                # Assuming graphrag_config is a mutable object (like a dict or OmegaConf dict)
                # The exact path might depend on GraphRagConfig structure
                try:
                    graphrag_config.local_search.temporal_filtering_enabled = override_temporal
                    print(f"  Overrode temporal_filtering_enabled to: {override_temporal}")
                except AttributeError:
                    print("  Warning: Could not override temporal_filtering_enabled in config.")
            # Repeat for relationship_timestamp_attribute if needed
            ```

### 8.3. Run `epbench` Benchmark

1.  **Prepare Benchmark:** Ensure your `epbench` dataset (`df_questions`, ground truth) contains questions that *specifically* test temporal reasoning (e.g., questions using "in 2023", "before event X", "after event Y").
2.  **Run `quickstart.py`:** Execute the script, ensuring the `--answering_kind` is `graphrag` and `--graphrag_index_dir` points to the index built with your modified GraphRAG code.
    ```bash
    python epbench/experiments/quickstart.py --answering_kind graphrag --graphrag_index_dir /path/to/your/graphrag_index_root
    ```
    (Add `--temporal_filtering_enabled False` or other overrides if implemented).
3.  **Analyze Results:** Examine the output metrics (F1 scores, potentially Kendall Tau if relevant). Compare the performance on temporal questions with results obtained without temporal filtering (by running with the override flag set to False, or by comparing to previous runs using unmodified GraphRAG).
