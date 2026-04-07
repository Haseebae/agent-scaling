import glob
import os
import pandas as pd
import sys
import yaml

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_scaling.agents.multiagent_utils.conversation import AgenSystemResult
from agent_scaling.utils import read_yaml

def parse_experiment_logs(root_dir):
    """
    Parses all agent_system_output.yaml files in a directory, extracts 
    token counts and other metrics, and returns a pandas DataFrame.
    """
    log_files = glob.glob(os.path.join(root_dir, '**', 'agent_system_output.yaml'), recursive=True)
    
    all_data = []

    for log_file in log_files:
        try:
            # Extract experiment identifiers from the file path
            path_parts = log_file.split(os.sep)
            experiment_name = path_parts[-7]
            model_name = path_parts[-5]
            task_id = path_parts[-2]

            # Load and parse the log file
            raw_data = read_yaml(log_file)
            if not raw_data:
                continue
            
            agent_result = AgenSystemResult(**raw_data)
            turn_counter = 1

            # Process lead agent conversation
            if agent_result.lead_agent_conversation:
                for message in agent_result.lead_agent_conversation.messages:
                    if message.litellm_message and message.litellm_message.usage:
                        
                        # Determine round number from tag
                        round_num = 0
                        if message.tag == 'planning':
                            round_num = 0
                        elif message.tag and message.tag.startswith('coordination_'):
                            try:
                                round_num = int(message.tag.split('_')[-1])
                            except ValueError:
                                round_num = -1 # Use -1 to signify a parsing error
                        
                        usage = message.litellm_message.usage
                        row = {
                            'experiment_name': experiment_name,
                            'model_name': model_name,
                            'task_id': task_id,
                            'agent_id': agent_result.lead_agent_conversation.agent_id,
                            'conversation_type': 'lead_agent',
                            'round_num': round_num,
                            'turn_num': turn_counter,
                            'prompt_tokens': usage.prompt_tokens,
                            'completion_tokens': usage.completion_tokens,
                            'total_tokens': usage.total_tokens,
                            'cost': message.cost,
                            'tag': message.tag
                        }
                        all_data.append(row)
                        turn_counter += 1

            # Process subagent conversations
            if agent_result.subagent_conversations:
                for agent_id, conv_history in agent_result.subagent_conversations.items():
                    for round_num, round_msgs in enumerate(conv_history.internal_comms):
                        turn_counter = 1
                        for turn_msg in round_msgs:
                            if turn_msg.litellm_message and turn_msg.litellm_message.usage:
                                usage = turn_msg.litellm_message.usage
                                row = {
                                    'experiment_name': experiment_name,
                                    'model_name': model_name,
                                    'task_id': task_id,
                                    'agent_id': agent_id,
                                    'conversation_type': 'subagent_internal',
                                    'round_num': round_num + 1,
                                    'turn_num': turn_counter,
                                    'prompt_tokens': usage.prompt_tokens,
                                    'completion_tokens': usage.completion_tokens,
                                    'total_tokens': usage.total_tokens,
                                    'cost': turn_msg.cost,
                                    'tag': None  # Internal comms don't have tags
                                }
                                all_data.append(row)
                                turn_counter += 1
        except Exception as e:
            print(f"Error parsing file {log_file}: {e}")

    return pd.DataFrame(all_data)

if __name__ == '__main__':
    output_dir = '../exp_outputs_test/plancraft-test'
    df = parse_experiment_logs(output_dir)
    
    output_path = './parsed_logs.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Successfully parsed {len(df)} messages from {len(df.task_id.unique())} instances.")
    print(f"Output saved to {output_path}")
    print("\nDataFrame Head:")
    print(df.head())
