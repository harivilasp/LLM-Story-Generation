import pandas as pd
import re

def read_and_process_file(filepath, pattern):
    with open(filepath) as f:
        data = f.readlines()
    data = [" ".join(i.split()) for i in data]
    data = [re.sub(pattern, '', i) for i in data]
    return data

def create_dataframe(stories_path, prompts_path, output_csv):
    pattern = r"\[\s*[A-Z]{2}\s*\]"
    stories = read_and_process_file(stories_path, pattern)
    prompts = read_and_process_file(prompts_path, pattern)
    df = pd.DataFrame({'stories': stories, 'prompts': prompts})
    df.to_csv(output_csv)

# Process training data
create_dataframe("data/writingprompts/train.wp_target", 
                 "data/writingprompts/train.wp_source", 
                 "data/train_df.csv")

# Process validation data
create_dataframe("data/writingprompts/valid.wp_target", 
                 "data/writingprompts/valid.wp_source", 
                 "data/valid_df.csv")
