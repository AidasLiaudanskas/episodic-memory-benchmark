# Default file paths to the data folder and the environment variable
from pathlib import Path
git_repo_filepath = '/filepath/to/gitrepo/episodic-memory-benchmark'
data_folder = Path(git_repo_filepath) / 'epbench' / 'data'
env_file = Path(git_repo_filepath) / '.env'

# Set global random seed for reproducibility
import random
import numpy as np
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Parsing the arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default=str(data_folder),
                    help='Path to the data folder')
parser.add_argument('--env_file', type=str, default=str(env_file),
                    help='Path to the .env file')
parser.add_argument('--book_nb_events', type=int, default=20,
                    help='Number of events in the book (20 for short and 200 for long, for the default experiment)')
parser.add_argument('--answering_kind', type=str, default='prompting',
                    help='Answering kind (e.g., prompting, rag, graphrag, ftuning)')
parser.add_argument('--answering_model_name', type=str, default='gpt-4o-mini-2024-07-18',
                    help='Answering model name')
parser.add_argument('--subset_fraction', type=float, default=1.0,
                    help='Fraction of questions to answer (between 0 and 1, default 1.0 answers all questions)')
parser.add_argument('--random_seed', type=int, default=RANDOM_SEED,
                    help=f'Random seed for reproducibility (default: {RANDOM_SEED})')
parser.add_argument('--graphrag_index_dir', type=str, default='',
                    help='Path to the GraphRAG index output directory (leave empty to auto-detect based on data_folder and nb_events)')

# Override the file paths and set any custom seed
args = parser.parse_args()

data_folder = Path(args.data_folder)
env_file = Path(args.env_file)

# Update random seed from arguments if provided
if args.random_seed != RANDOM_SEED:
    RANDOM_SEED = args.random_seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"Using random seed: {RANDOM_SEED}")

# Step 1: generating the synthetic episodic memory dataset

## Configuration (here, default short book with 20 events)
book_parameters = {
  'indexing': 'default', 
  'nb_summaries': 0
  }
prompt_parameters = {
  'nb_events': args.book_nb_events, 
  'name_universe': 'default', 
  'name_styles': 'default', 
  'seed': 0, 
  'distribution_events': {
    'name': 'geometric', 
    'param': 0.1
    }
  }
model_parameters = {
  'model_name': 'claude-3-5-sonnet-20240620', 
  'max_new_tokens': 4096, 
  'itermax': 10
  }

## Generation (generate the book, then compute the ground truth QAs)
from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
my_benchmark = BenchmarkGenerationWrapper(
  prompt_parameters, model_parameters, book_parameters, data_folder, env_file, rechecking=False)

# Step 2: predicting the answers given the document and the questions

## Configuration
answering_parameters = {
  'kind': args.answering_kind, 
  'model_name': args.answering_model_name, 
  'max_new_tokens': 4096, 
  'sleeping_time': 0, 
  'policy': 'remove_duplicates',
  'subset_fraction': args.subset_fraction,
  'random_seed': RANDOM_SEED
  }

# ADDED: Conditionally add graphrag_index_dir to parameters, constructing default if needed
if args.answering_kind == 'graphrag':
    graphrag_dir_arg = args.graphrag_index_dir
    if not graphrag_dir_arg: # If the argument was empty, use the benchmark output folder
        # Use the benchmark output folder as the root for the GraphRAG index
        # Assumes graphrag index was built with this folder as its root,
        # containing settings.yaml and the output/ subfolder.
        graphrag_dir = my_benchmark.get_benchmark_dirpath()
        print(f"--graphrag_index_dir not specified, using benchmark output folder as GraphRAG root: {graphrag_dir}")
    else:
        graphrag_dir = Path(graphrag_dir_arg) # Ensure it's a Path object

    # Store the resolved, absolute path string
    answering_parameters['graphrag_index_dir'] = str(graphrag_dir.resolve())

    # Ensure the directory exists (basic check)
    if not graphrag_dir.is_dir():
        print(f"Warning: Constructed or provided GraphRAG index root directory does not exist: {graphrag_dir}")
        # Consider raising an error here if the directory MUST exist before proceeding
        # raise FileNotFoundError(f"GraphRAG index root directory not found: {graphrag_dir}")

## Prediction (generate answers, then evaluate them)
from epbench.src.evaluation.evaluation_wrapper import EvaluationWrapper
my_evaluation = EvaluationWrapper(my_benchmark, answering_parameters, data_folder, env_file)

# Step 3: extract the performance results - only if we're using all questions
if args.subset_fraction < 1.0 or args.book_nb_events not in [20, 200, 2000]:
    # For subset evaluation or non-standard chapter counts, calculate results directly
    import pandas as pd
    import numpy as np
    
    if args.book_nb_events not in [20, 200, 2000] and args.subset_fraction == 1.0:
        print(f"\nNon-standard chapter count ({args.book_nb_events}). Using direct evaluation rather than precomputed_results.")
    
    print("\nEvaluation Results Summary:")
    print("===========================")
    
    # Get the evaluated data
    df_eval = my_evaluation.df_generated_evaluations
    
    if not df_eval.empty:
        # Calculate bins for items in correct answer, similar to extract_groups
        bins_count = [0, 1, 2, 3, 6, np.inf]
        labels_count = ['0', '1', '2', '3-5', '6+']
        df_eval['bins_items_correct_answer'] = pd.cut(df_eval['n_chapters_correct_answer'], 
                                                    bins=bins_count, 
                                                    include_lowest=True, 
                                                    right=False, 
                                                    labels=labels_count)
        
        # Group by bins and calculate statistics
        grouped = df_eval.groupby('bins_items_correct_answer')
        results = []
        
        for name, group in grouped:
            count = len(group)
            # Calculate F1 score mean and std
            if 'f1_score_lenient' in group.columns:
                f1_mean = group['f1_score_lenient'].mean()
                f1_std = group['f1_score_lenient'].std()
                results.append({
                    'bins_items_correct_answer': name,
                    'count': count,
                    f'(prompting, {args.answering_model_name}, n/a)': f"{f1_mean:.2f}±{f1_std:.2f}"
                })
        
        # Create results dataframe and display
        if results:
            df_results = pd.DataFrame(results)
            print(df_results.to_string(index=False))
        
        # Print Kendall Tau summaries if available
        if hasattr(my_evaluation, 'kendall_summaries_for_this_experiment') and not my_evaluation.kendall_summaries_for_this_experiment.empty:
            print("\nChronological Ordering Results:")
            print(my_evaluation.kendall_summaries_for_this_experiment)
    
    print('Ended successfully - Skipped full results processing due to subset_fraction < 1.0')
else:
    ## Configuration for full evaluation
    experiments = [{
      'book_nb_events': args.book_nb_events, 
      'book_model_name': 'claude-3-5-sonnet-20240620',
      'answering_kind': args.answering_kind, 
      'answering_model_name': args.answering_model_name,
      'answering_embedding_chunk': 'n/a'
      },
    ]
    all_benchmarks = {f'benchmark_claude_default_{args.book_nb_events}': my_benchmark}

    ## Results
    from epbench.src.evaluation.precomputed_results import get_precomputed_results
    df = get_precomputed_results(experiments, env_file, data_folder, all_benchmarks)
    df

    from epbench.src.results.average_groups import extract_groups
    # select the book of interest (either 20 or 200)
    nb_events = args.book_nb_events
    # select the elements to group
    relative_to = ['get', 'bins_items_correct_answer']
    # group the results according to `relative_to`
    df_results = extract_groups(df, nb_events, relative_to)
    # further filtering by selecting only the simple recall questions
    df_results = df_results[df_results['get'] == 'all'].drop('get', axis = 1)

    print(df_results)
    print('Ended successfully')
