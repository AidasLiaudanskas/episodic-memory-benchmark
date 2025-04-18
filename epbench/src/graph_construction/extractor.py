from typing import List, Dict, Any, Tuple
from openai import OpenAI
from epbench.src.models.settings_wrapper import SettingsWrapper
import json
import traceback # For better error logging
import os
from dotenv import load_dotenv

# Define expected entity and relationship types (customize as needed)
ENTITY_TYPES = ["Person", "Location", "Time", "Event", "Object", "Organization"]
RELATIONSHIP_TYPES = ["PRECEDES", "FOLLOWS", "LOCATED_AT", "PARTICIPATED_IN", "USES", "ASSOCIATED_WITH"]

def build_extraction_prompt(text_chunk: str) -> str:
    """Creates the prompt for the LLM to extract entities and relationships."""
    # Simplified prompt without complex inline examples
    prompt = f"""
Analyze the following text chunk from a narrative:
-------
{text_chunk}
-------

Identify the key entities and relationships within this text. 
Entities should be categorized into one of the following types: {list(ENTITY_TYPES)}.
Relationships should describe connections between entities and be one of the following types: {list(RELATIONSHIP_TYPES)}.

Output the results as a valid JSON object containing two keys: "entities" and "relationships".

The value for "entities" should be a list of JSON objects. Each entity object must have:
- "id": A unique identifier for the entity within this chunk (e.g., the entity name or a concept). Ensure IDs are consistent if an entity appears multiple times.
- "type": One of the allowed entity types.
- "name": The name or description of the entity as it appears in the text.

The value for "relationships" should be a list of JSON objects. Each relationship object must have:
- "source": The id of the source entity.
- "target": The id of the target entity.
- "type": One of the allowed relationship types.
- "description": (Optional) A brief phrase from the text describing the relationship.

Provide *only* the JSON object in your response. Do not include any explanations or introductory text before or after the JSON.
"""
    return prompt

def extract_entities_relationships_llm(
    text_chunks: List[str],
    env_file: str,
    model_name: str = "gpt-3.5-turbo", # Or specify another model like gpt-4
    max_retries: int = 3,
    request_timeout: int = 120,
    # Add parameters for custom prompts if needed
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Extracts entities and relationships from a list of text chunks using an LLM.

    Args:
        text_chunks: A list of strings, where each string is a text chunk.
        env_file: Path to the .env file containing the OPENAI_API_KEY.
        model_name: The OpenAI model to use for extraction.
        max_retries: Maximum number of retries for API calls.
        request_timeout: Timeout for the API request in seconds.

    Returns:
        A list of tuples. Each tuple corresponds to a text chunk and contains:
        - A list of extracted entity dictionaries.
        - A list of extracted relationship dictionaries.
        Returns empty lists for a chunk if extraction fails for that chunk.
    """
    load_dotenv(dotenv_path=env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY not found in {env_file}. Please ensure it's set.")

    client = OpenAI(api_key=api_key)
    
    results = []
    print(f"Starting extraction for {len(text_chunks)} chunks using {model_name}...")

    for i, chunk in enumerate(text_chunks):
        print(f"  Processing chunk {i+1}/{len(text_chunks)}...")
        
        # --- Define the Prompt ---
        # This is a crucial part. Needs careful design and testing.
        # We want structured JSON output.
        prompt = f"""
Extract entities and relationships from the following text.
Identify key entities (people, locations, objects, concepts) and relationships between them (e.g., interactions, locations, properties).

Rules:
- Output MUST be a valid JSON object.
- The JSON object must have two keys: "entities" and "relationships".
- "entities" should be a list of objects, each with "id" (unique identifier, prefer noun phrases), "type" (e.g., Person, Location, Object, Event), and "name" (the text span).
- "relationships" should be a list of objects, each with "source" (entity id), "target" (entity id), "type" (e.g., LOCATED_AT, INTERACTED_WITH, PART_OF, PRECEDES), and optionally "description".
- Use the entity "id" values for "source" and "target" in relationships. Choose descriptive IDs.
- If no entities or relationships are found, return empty lists for the respective keys.

Text:
"{chunk}"

JSON Output:
"""
        # --- End Prompt Definition ---

        entities = []
        relationships = []
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert entity and relationship extractor. Your output must be valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2, # Lower temperature for more deterministic output
                    max_tokens=1000, # Adjust as needed based on expected output size
                    timeout=request_timeout,
                    response_format={"type": "json_object"}, # Request JSON output
                )
                
                content = response.choices[0].message.content
                if content:
                    data = json.loads(content)
                    entities = data.get("entities", [])
                    relationships = data.get("relationships", [])
                    print(f"    Successfully extracted {len(entities)} entities and {len(relationships)} relationships.")
                    break # Success, exit retry loop
                else:
                     print(f"    Attempt {attempt + 1}: Received empty content.")
                
            except json.JSONDecodeError as json_e:
                 print(f"    Attempt {attempt + 1}: Failed to decode JSON response: {json_e}")
                 print(f"    Received content: {content}")
            except Exception as e:
                print(f"    Attempt {attempt + 1}: API call failed: {e}")
                if attempt == max_retries - 1:
                     print(f"  Failed to extract from chunk {i+1} after {max_retries} attempts.")
                     # Append empty results for this chunk on final failure
                     # results.append(([], [])) # Let the outer loop handle appending

        results.append((entities, relationships)) # Append results for the chunk

    print("Extraction finished.")
    return results

# Example usage
if __name__ == '__main__':
    print("Running extractor example...")
    sample_chapters = [
        "Chapter 1: Alice walked to the old library. It was Tuesday. She carried a red book.",
        "Chapter 2: Later, Bob met Alice at the park near the library. They discussed the book.",
        ""
    ]
    # IMPORTANT: Replace with the actual path to your .env file
    # Assumes .env is two levels up from epbench/src/graph_construction/
    env_file_path = '../../.env' 

    try:
        # Use a cheaper/faster model for testing if needed
        # test_model = "gpt-3.5-turbo"
        test_model = "gpt-4o-mini-2024-07-18" 
        extracted_data_list = extract_entities_relationships_llm(
            sample_chapters, 
            env_file=env_file_path, 
            model_name=test_model
        )
        
        for i, (entities, relationships) in enumerate(extracted_data_list):
            print(f"\n--- Chunk {i+1} Results ---")
            if not entities and not relationships:
                print("  (No data extracted or error occurred)")
            else:
                print(f"  Entities ({len(entities)}): {json.dumps(entities, indent=2)}")
                print(f"  Relationships ({len(relationships)}): {json.dumps(relationships, indent=2)}")

    except Exception as e:
        print(f"\nExample failed: {e}")
        print(traceback.format_exc())
        print("Ensure your OpenAI API key is set in the .env file and the path is correct.") 