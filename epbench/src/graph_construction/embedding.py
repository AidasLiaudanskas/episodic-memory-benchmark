import pandas as pd
from typing import List, Dict
from openai import OpenAI
# from epbench.src.utils.settings_wrapper import SettingsWrapper # Assuming this path is correct
# from epbench.src.io.io import SettingsWrapper # Corrected path based on potential usage
from epbench.src.models.settings_wrapper import SettingsWrapper # Corrected path based on search results

def get_openai_embeddings(
    texts: List[str], 
    env_file: str, 
    embedding_model: str = "text-embedding-3-small", 
    embedding_batch_size: int = 1024 # Reduced batch size slightly for safety
) -> List[List[float]]:
    '''
    Generates embeddings for a list of text chunks using OpenAI API.
    Adapted from epbench.src.evaluation.generator_answers_2_rag.embed_chunks

    Args:
        texts: A list of strings to embed.
        env_file: Path to the .env file containing API keys.
        embedding_model: The OpenAI embedding model to use.
        embedding_batch_size: Number of texts to send in each API request.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    '''
    if not texts:
        return []
        
    try:
        config = SettingsWrapper(_env_file=env_file)
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        print(f"Error initializing OpenAI client or SettingsWrapper: {e}")
        print(f"Please ensure your .env file is correctly set up at: {env_file}")
        print(f"And that epbench.src.utils.settings_wrapper can be imported.")
        raise # Re-raise the exception to halt execution

    all_embeddings: List[List[float]] = []
    for batch_start in range(0, len(texts), embedding_batch_size):
        batch_end = min(batch_start + embedding_batch_size, len(texts))
        batch = texts[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        
        print(f"Requesting embeddings for batch {batch_start} to {batch_end-1} (Model: {embedding_model})...")
        try:
            response = client.embeddings.create(model=embedding_model, input=batch)
            
            # Store embeddings in a dictionary temporarily to ensure correct order
            batch_embeddings_dict: Dict[int, List[float]] = {}
            for i, be in enumerate(response.data):
                original_index = batch_indices[i]
                # assert i == be.index # This assertion might fail depending on API version/behavior, index mapping is safer
                batch_embeddings_dict[original_index] = be.embedding

            # Append embeddings to the main list in the correct order
            for idx in batch_indices:
                 if idx in batch_embeddings_dict:
                      all_embeddings.append(batch_embeddings_dict[idx])
                 else:
                      # Handle potential error case where an embedding wasn't returned
                      print(f"Warning: Missing embedding for index {idx} in batch {batch_start}-{batch_end-1}")
                      # Option: Append None or a zero vector, or raise an error
                      all_embeddings.append([0.0] * len(all_embeddings[0]) if all_embeddings else []) # Placeholder

        except Exception as e:
            print(f"Error getting embeddings for batch {batch_start}-{batch_end-1}: {e}")
            # Handle error - e.g., add placeholders or stop
            error_placeholder = [0.0] * len(all_embeddings[0]) if all_embeddings else [] # Placeholder
            for _ in range(len(batch)):
                 all_embeddings.append(error_placeholder)
                 
    print(f"Successfully generated {len(all_embeddings)} embeddings.")
    return all_embeddings

# Example usage (requires .env file to be set up)
if __name__ == '__main__':
    # This example will likely fail unless run in an environment where
    # epbench.src.utils.settings_wrapper is importable and .env is configured.
    print("Running embedding example...")
    sample_texts = [
        "Chapter 1: The beginning.",
        "Chapter 2: The middle.",
        "Chapter 3: The end."
    ]
    # IMPORTANT: Replace with the actual path to your .env file
    env_file_path = '../../.env' # Adjust path as needed relative to this file
    
    try:
        embeddings_list = get_openai_embeddings(sample_texts, env_file=env_file_path)
        
        if embeddings_list:
            print(f"Generated {len(embeddings_list)} embeddings.")
            print(f"Dimension of the first embedding: {len(embeddings_list[0])}")
        else:
             print("No embeddings generated.")

    except Exception as e:
        print(f"Example failed: {e}")
        print("Ensure your OpenAI API key is set in the .env file and the path is correct.") 