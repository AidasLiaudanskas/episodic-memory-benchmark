from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from epbench.src.evaluation.generator_answers_1_prompting import generate_answers_func, generate_evaluation_func, generate_chronological_func
from epbench.src.evaluation.scoring_answers import update_policy_of_evaluation_to
import numpy as np
# for rag
from epbench.src.generation.generate_3_secondary_entities import count_tokens
from epbench.src.evaluation.generator_answers_2_rag import embed_chunks
# --- ADDED: Import for GraphRAG ---
from epbench.src.evaluation.generator_answers_4_graphrag import generate_answers_graphrag
# --- End Added Import ---
from epbench.src.io.io import answer_dirpath_func, import_list, export_list, export_jsonl
import pandas as pd
import ast
# for ftuning
from epbench.src.evaluation.generator_answers_3_ftuning import upload_ftuning_input, retrieve_fileid
from openai import OpenAI
from epbench.src.models.settings_wrapper import SettingsWrapper
from scipy.stats import kendalltau
from pathlib import Path

class EvaluationWrapper:
    def __init__(
            self,
            my_benchmark: BenchmarkGenerationWrapper,
            answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'sleeping_time': 15, 'policy': 'original'},
            data_folder = '/repo/to/git/main/epbench/data',
            env_file = '/repo/to/git/main/.env'):
        
        # save the input
        self.my_benchmark = my_benchmark
        self.data_folder = data_folder
        self.env_file = env_file
        self.policy = answering_parameters['policy']
        
        # Print subset info if applicable
        if 'subset_fraction' in answering_parameters and answering_parameters['subset_fraction'] < 1.0:
            print(f"Using subset_fraction = {answering_parameters['subset_fraction']}")

        # Store answering parameters for later use if needed
        self.answering_parameters = answering_parameters

        # generated answers
        if answering_parameters['kind'] == 'prompting':
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file)
        elif answering_parameters['kind'] == 'rag':
            ## [embedding code] -- all this paragraph to save the embedding
            # answering_parameters = {'kind': 'rag', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'sleeping_time': 15, 'embedding_chunk': 'chapter', 'embedding_model': "text-embedding-3-small", 'embedding_batch_size': 2048}
            # alternative embedding_model: "text-embedding-ada-002"
            self.my_chunks = my_benchmark.chunk_book(split = answering_parameters['embedding_chunk'])
            self.chunk_with_max_tokens = max([count_tokens(x) for x in self.my_chunks])
            self.chunk_number = len(self.my_chunks)

            prompt_parameters = my_benchmark.prompt_parameters
            model_parameters = my_benchmark.model_parameters
            book_parameters = my_benchmark.book_parameters
            nb_chapters = my_benchmark.nb_chapters()
            nb_tokens = my_benchmark.nb_tokens()
            answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            embedding_filepath = answer_dirpath / "embedding.csv"
            if not embedding_filepath.is_file():
                my_embedding = embed_chunks(self.my_chunks, answering_parameters, env_file)
                answer_dirpath.mkdir(parents=True, exist_ok=True)
                my_embedding.to_csv(embedding_filepath, index=False)
                print(f"chunked into {self.chunk_number} chunks, the largest containing {self.chunk_with_max_tokens} tokens")
                print(my_embedding)
            self.my_embedding = pd.read_csv(embedding_filepath)
            self.my_embedding['embedding'] = self.my_embedding['embedding'].apply(ast.literal_eval)
            ## [\embedding code] -- the rest as for the 'prompting' kind
            
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file, self.my_embedding)
        elif answering_parameters['kind'] == 'graphrag':
            print("--- Handling 'graphrag' answering kind ---")
            
            # 1. Get and validate the GraphRAG index path from answering_parameters
            try:
                # Use the path provided by quickstart.py
                graphrag_index_path = Path(self.answering_parameters['graphrag_index_dir'])
                print(f"Target GraphRAG index directory: {graphrag_index_path}")
                
                # Use Path object method for checking existence
                if not graphrag_index_path.is_dir():
                    raise FileNotFoundError(f"GraphRAG index directory not found at specified location: {graphrag_index_path}")
                    
            except KeyError:
                 print("ERROR: 'graphrag_index_dir' not found in answering_parameters.")
                 print("       Please ensure it's correctly passed from quickstart.py.")
                 raise # Re-raise to stop execution
            except FileNotFoundError as e:
                 print(f"ERROR: {e}")
                 raise # Re-raise to stop execution
            
            # MOVED UP: Get the full QA DataFrame *before* calling the generator
            df_qa_full = self.my_benchmark.get_df_qa().reset_index().rename(columns={'index': 'q_idx'}) # Ensure q_idx column exists

            # 3. Call the GraphRAG answer generation function, passing the index path and questions
            print("Generating answers using GraphRAG...")
            # generate_answers_graphrag now returns the full DataFrame
            # Assign directly to self.df_generated_answers
            self.df_generated_answers = generate_answers_graphrag(
                df_questions=df_qa_full, # Pass the questions DataFrame
                graph_components_dir=graphrag_index_path, # Pass the determined path
                env_file=env_file,
                my_benchmark=my_benchmark,
                answering_parameters=self.answering_parameters, # Pass the whole dict
                data_folder=data_folder, # Needed for constructing answer file paths inside generator
                subset_fraction=self.answering_parameters.get('subset_fraction', 1.0),
                random_seed=self.answering_parameters.get('random_seed', 42)
            )
            # REMOVED: The raw dataframe and merge step are no longer needed
            # df_generated_answers_raw = generate_answers_graphrag(...)
            # print(f"Raw answers generated: {len(df_generated_answers_raw)} rows")
            
            # 4. Merge raw answers (q_idx, llm_answer) with the full ground truth QA dataframe
            # This ensures the final DataFrame has all necessary columns for evaluation steps
            # REMOVED: df_qa_full definition moved above
            # df_qa_full = self.my_benchmark.get_df_qa().reset_index().rename(columns={'index': 'q_idx'}) # Ensure q_idx column exists
            
            # REMOVED: Merge step is now handled inside generate_answers_graphrag
            # print(f"Merging {len(df_generated_answers_raw)} answers with {len(df_qa_full)} ground truth questions...")
            # self.df_generated_answers = pd.merge(
            #     df_qa_full,
            #     df_generated_answers_raw[['q_idx', 'llm_answer']], # Select only needed cols
            #     on='q_idx',
            #     how='left' # Keep all original questions, add answers where available
            # )
            print("Finished GraphRAG answer generation.")
            # Clear the loaded pipeline from memory if desired (optional)
            # del self.graphrag_pipeline 
        elif answering_parameters['kind'] == 'ftuning':
            ## [fine tuning input data code] -- all this paragraph for uploading the jsonl
            
            # 1. build jsonl
            prompt_parameters = my_benchmark.prompt_parameters
            model_parameters = my_benchmark.model_parameters
            book_parameters = my_benchmark.book_parameters
            nb_chapters = my_benchmark.nb_chapters()
            nb_tokens = my_benchmark.nb_tokens()
            ftuning_input_data_policy = answering_parameters['ftuning_input_data_policy']
            if ftuning_input_data_policy == 'single':
                answer_in_one_chapter_only = True # only select questions with single answers
                ftuning_input = my_benchmark.build_fine_tuning_jsonl(answer_in_one_chapter_only)
            elif ftuning_input_data_policy == 'all':
                answer_in_one_chapter_only = False # overfit on all possible questions
                ftuning_input = my_benchmark.build_fine_tuning_jsonl(answer_in_one_chapter_only)
            answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            jsonl_filename = f"ftuning_{ftuning_input_data_policy}_{nb_tokens}.jsonl"
            ftuning_input_filepath = answer_dirpath / jsonl_filename
            if not ftuning_input_filepath.is_file():
                answer_dirpath.mkdir(parents=True, exist_ok=True)
                export_jsonl(ftuning_input, ftuning_input_filepath)
            self.ftuning_input = ftuning_input
            config = SettingsWrapper(_env_file = env_file)

            # 2. uploading
            if answering_parameters['ftuning_need_upload']:
                # [client]
                is_existing_client = True
                try:
                    self.client
                except AttributeError:
                    is_existing_client = False
                if not is_existing_client:
                    self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                # [\client]
                upload_ftuning_input(self.client, ftuning_input_filepath)

            # 3. get the corresponding created fileid
            ftuning_input_id_filepath = answer_dirpath / f"ftuning_{ftuning_input_data_policy}.id"
            if not ftuning_input_id_filepath.is_file():
                answer_dirpath.mkdir(parents=True, exist_ok=True)
                # [client]
                is_existing_client = True
                try:
                    self.client
                except AttributeError:
                    is_existing_client = False
                if not is_existing_client:
                    self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                # [\client]
                ftuning_input_id = retrieve_fileid(self.client, jsonl_filename)
                export_list(ftuning_input_id, ftuning_input_id_filepath)
                print(f"json `{jsonl_filename}` with file id `{self.ftuning_input_id}` for the `{ftuning_input_data_policy}` policy")
            self.ftuning_input_id = import_list(ftuning_input_id_filepath)
            
            # 4. create a fine tuning job
            if answering_parameters['ftuning_need_actual_tune']:
                # [client]
                is_existing_client = True
                try:
                    self.client
                except AttributeError:
                    is_existing_client = False
                if not is_existing_client:
                    self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                # [\client]
                self.client.fine_tuning.jobs.create(
                    training_file=self.ftuning_input_id,
                    model=answering_parameters['model_name'],
                    hyperparameters={
                    "batch_size": answering_parameters['batch_size'],
                    "learning_rate_multiplier": answering_parameters['learning_rate_multiplier'],
                    "n_epochs": answering_parameters['n_epochs']
                    #max_tokens=max_tokens,
                    #training_file=os.path.abspath(training_file),
                    #validation_file=os.path.abspath(validation_file),
                    }
                )
                print('ongoing jobs')
                print(self.client.fine_tuning.jobs.list(limit=10))
                print('for cancelling, please use `object.cancel_job(corresponding_ftjob_as_above)`')

            # 5. answer the questions
            answering_parameters['model_name'] = answering_parameters['fine_tuned_model_name']
             #print(answering_parameters['model_name'])

            # NOTE: ftuning currently doesn't support subset_fraction, so we generate all answers
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file, my_embedding=None)

        else:
            raise ValueError('unknown "kind", should be "prompting", "rag" or "ftuning"')
        
        # Print answer statistics
        if self.df_generated_answers.empty:
            print("\nWarning: No answers were generated. Skipping evaluation and chronological steps.")
            self.df_generated_evaluations = pd.DataFrame()
            self.df_generated_chronological = pd.DataFrame()
            self.kendall_summaries_for_this_experiment = pd.DataFrame()
        else:
            total_questions_in_benchmark = len(my_benchmark.get_df_qa())
            # Filter out rows where answers were not generated (None or specific placeholders)
            # Common placeholders: "Not Processed (Subset)", "Error reading cached answer", None
            placeholders = ["Not Processed (Subset)", "Error reading cached answer"]
            df_answers_to_evaluate = self.df_generated_answers[
                self.df_generated_answers['llm_answer'].notna() &
                ~self.df_generated_answers['llm_answer'].isin(placeholders)
            ].copy()
 
            answered_questions = len(df_answers_to_evaluate)
            if 'subset_fraction' in answering_parameters and answering_parameters['subset_fraction'] < 1.0:
                 total_questions_expected = len(self.df_generated_answers) # Should reflect the subset size if merge worked
                 print(f"\nGenerated answers for {answered_questions}/{total_questions_expected} questions in the subset.")
            else:
                 print(f"\nGenerated answers for {answered_questions}/{total_questions_in_benchmark} total questions ({answered_questions/total_questions_in_benchmark*100:.1f}%)")
 
            if answered_questions == 0:
                 print("Warning: No valid answers found to evaluate. Skipping evaluation and chronological steps.")
                 self.df_generated_evaluations = pd.DataFrame()
                 self.df_generated_chronological = pd.DataFrame()
                 self.kendall_summaries_for_this_experiment = pd.DataFrame()
                 return # Exit init early
              
            # generated evaluation (given answers)
            # Pass only the successfully answered questions to the evaluation function
            print(f"\nStarting evaluation for {answered_questions} answered questions...")
            df_generated_evaluations_subset = generate_evaluation_func(my_benchmark, df_answers_to_evaluate, answering_parameters, data_folder, env_file)
            # possibly with a different policy for the final evaluation
            self.df_generated_evaluations = update_policy_of_evaluation_to(df_generated_evaluations_subset, self.policy)
              
            # Print evaluation statistics
            total_evaluations = len(self.df_generated_evaluations)
            print(f"Finished evaluation. Generated evaluations for {total_evaluations}/{answered_questions} answered questions.")
            if total_evaluations < answered_questions:
                 print(f"  Warning: Some answered questions ({answered_questions - total_evaluations}) might have failed during evaluation.")
  
            # generated chronological (given evaluation)
            # Pass the evaluated subset to the chronological function
            print("\nStarting chronological evaluation...")
            df_generated_chronological = generate_chronological_func(my_benchmark, self.df_generated_evaluations, answering_parameters, data_folder, env_file)
            self.df_generated_chronological = df_generated_chronological
            
            # Print chronological statistics
            if len(df_generated_chronological) > 0:
                num_chrono_evaluated = len(df_generated_chronological)
                num_chrono_possible = len(self.df_generated_evaluations[self.df_generated_evaluations['get'] == 'chronological']) # Count possible chrono qs in evaluated subset
                print(f"Finished chronological evaluation. Generated evaluations for {num_chrono_evaluated}/{num_chrono_possible} applicable questions.")
              
            self.kendall_summaries_for_this_experiment = self.compute_kendall_summarise(df_generated_chronological, verbose = False)

    def cancel_job(self, ftjob_id = 'ftjob-wjldwdkjw0eiw'):
        self.client.fine_tuning.jobs.cancel(ftjob_id)
        print("cancelled")

    def get_pretty_summary_relative_to(self, my_column = 'q_idx', metric = 'f1_score_lenient', sorting = False, filter_dict = {}):
        df = self.df_generated_evaluations

        if 'bins_items_correct_answer' in my_column:
            bins_count = [0, 1, 2, 3, 6, np.inf]
            labels_count = ['0', '1', '2', '3-5', '6+']
            df['bins_items_correct_answer'] = pd.cut(df['n_chapters_correct_answer'], bins=bins_count, include_lowest=True, right=False, labels=labels_count)
            #print(df['bins_items_correct_answer'])
        
        if 'bins_items_correct_answer_few' in my_column:
            bins_count = [0, 2, np.inf]
            labels_count = ['0-1', '2+']
            df['bins_items_correct_answer_few'] = pd.cut(df['n_chapters_correct_answer'], bins=bins_count, include_lowest=True, right=False, labels=labels_count)

        if 'cue_size' in my_column:
            df['cue_size'] = [4-elem.count('*') for elem in df['cue']]

        for column, value in filter_dict.items():
            df = df[df[column] == value]

        if my_column == '':
            result = df[[metric]].copy()
            result['count'] = 1
            result = result[['count', metric]]
            return result

        result = df.groupby(my_column, observed = False).agg({
            metric: ['mean', 'std', 'count']
        })
        result.columns = ['f1_score_mean', 'f1_score_std', 'count']
        result = result.reset_index()
        result[metric] = result.apply(lambda row: f"{row['f1_score_mean']:.2f}±{row['f1_score_std']:.2f}", axis=1)
        if sorting:
            result = result.sort_values(by='f1_score_mean', ascending=True)
        result = result.drop(columns=['f1_score_mean', 'f1_score_std'])
        # Reset the index to make 'n_items' a column
        result = result.reset_index(drop=True)
        return result
    
    def remove_duplicates_and_negative_one(self, lst):
        seen = set()
        return [x for x in lst if x != -1 and not (x in seen or seen.add(x))]

    def process_lists_and_compute_kendall_tau(self, l1, l2):
        # Step 1: Keep only elements that are in both lists
        common_elements = list(set(l1) & set(l2))
        nb_matches = len(common_elements)
        
        # Create new lists with only common elements, preserving original order
        result_l1 = [x for x in l1 if x in common_elements]
        result_l2 = [x for x in l2 if x in common_elements]
        
        # Step 2: Compute Kendall tau (discard the p-value)
        tau, _ = kendalltau(
            [result_l1.index(x) for x in common_elements],
            [result_l2.index(x) for x in common_elements]
        )
        
        return float(tau), nb_matches

    def compute_kendall_summarise(self, df_generated_chronological, verbose = True):
        chrono = df_generated_chronological

        chrono['filtered_predicted_indexes'] = [self.remove_duplicates_and_negative_one(lst) for lst in chrono['predicted_indexes']]
        chrono['groundtruth_indexes_length'] = [len(x) for x in chrono['groundtruth_indexes']]

        chrono['tau'] = [self.process_lists_and_compute_kendall_tau(l1, l2)[0] for (l1, l2) in zip(chrono['filtered_predicted_indexes'], chrono['groundtruth_indexes'])]
        chrono['nb_matches'] = [self.process_lists_and_compute_kendall_tau(l1, l2)[1] for (l1, l2) in zip(chrono['filtered_predicted_indexes'], chrono['groundtruth_indexes'])]

        # all the samples in which the chronological order can be tested
        chrono_larger_than_one = chrono[chrono['groundtruth_indexes_length'] > 1]
        N = len(chrono_larger_than_one)

        chrono_gt_larger_than_one_with_total_match = chrono_larger_than_one[chrono_larger_than_one['nb_matches'] == chrono_larger_than_one['groundtruth_indexes_length']]
        count_of_total_match = sum(chrono_larger_than_one['nb_matches'] == chrono_larger_than_one['groundtruth_indexes_length']) # % total match
        percentage_of_total_match = 100*round(count_of_total_match/N, 2)
        tau_over_total_match = float(chrono_gt_larger_than_one_with_total_match['tau'].mean())
        sd_tau_over_total_match = float(chrono_gt_larger_than_one_with_total_match['tau'].std())

        chrono_gt_larger_than_one_with_match_greater_than_one = chrono_larger_than_one[chrono_larger_than_one['nb_matches'] > 1]
        count_of_match_greater_than_1 = sum(chrono_larger_than_one['nb_matches'] > 1) # % match > 1
        percentage_of_match_greater_than_1 = 100*round(count_of_match_greater_than_1/N, 2)
        tau_over_match_greater_than_one = float(chrono_gt_larger_than_one_with_match_greater_than_one['tau'].mean())
        sd_tau_over_match_greater_than_one = float(chrono_gt_larger_than_one_with_match_greater_than_one['tau'].std())

        if verbose:
            print(f"For the {count_of_total_match} samples with exact match between pred and gt sets (among the {N} with #gt>1, i.e. occurring for {percentage_of_total_match}%), the kendall tau average is {round(tau_over_total_match,2)}±{round(sd_tau_over_total_match,2)}.")
            print(f"For the {count_of_match_greater_than_1} samples with partial match >1 between pred and gt sets (among the {N} with #gt>1, i.e. occurring for {percentage_of_match_greater_than_1}%), the kendall tau average is {round(tau_over_match_greater_than_one,2)}±{round(sd_tau_over_match_greater_than_one,2)}.")

        kendall_summaries_for_this_experiment = pd.DataFrame({
            '#gt_with_len_2+': N, 
            '#exact_match_set_gt_with_pred': count_of_total_match, 
            '%_exact_match_set_gt_with_pred': f"{round(percentage_of_total_match)}%", 
            'tau_exact_match_set_gt_with_pred': f"{round(tau_over_total_match,2)}±{round(sd_tau_over_total_match,2)}",
            '#partial_match_set_gt_with_pred': count_of_match_greater_than_1, 
            '%_partial_match_set_gt_with_pred': f"{round(percentage_of_match_greater_than_1)}%", 
            'tau_partial_match_set_gt_with_pred': f"{round(tau_over_match_greater_than_one,2)}±{round(sd_tau_over_match_greater_than_one,2)}"
            }, index = [0])
        return kendall_summaries_for_this_experiment
