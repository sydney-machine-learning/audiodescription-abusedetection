import pandas as pd

import os
import logging
import re

import utils


sub_dir = os.path.join('data', 'subtitles')
time_sep = '-->'


def extract_subs():
    subs_file_list = [os.path.join(sub_dir, x) for x in os.listdir(sub_dir) if not x.endswith('.zip')]
    movie_subs_dict = {}

    for sub_file in subs_file_list:
        with open(sub_file, 'r', encoding='utf-8-sig') as fileobj:
            raw_timestamps = []
            raw_dialogue = []
            counter = -1
            
            # Separate lines of dialogue from timestamps into raw lists
            for raw_line in fileobj.readlines():
                line = raw_line.strip()
                
                if re.match('^\d{1,}$', line):
                    counter += 1
                    raw_dialogue.append('')
                elif time_sep in line:
                    raw_timestamps.append(line)
                elif line != '':
                    raw_dialogue[counter] += (' ' + line)
        
        # Create dataframe
        if 'subtitle' in raw_dialogue[0].lower():
            raw_dialogue = raw_dialogue[1:]
            raw_timestamps = raw_timestamps[1:]
        
        subs_df = pd.DataFrame({'raw_dialogue': raw_dialogue, 'raw_time_str': raw_timestamps})
        
        # Replace commas with decimals to ensure conversion goes smoothly
        subs_df['raw_time_str'] = subs_df['raw_time_str'].str.replace(',', '.')
        
        # Remove new line characters on ends of dialogue (current method still keeps 'internal' new line characters)
        subs_df['raw_dialogue'] = subs_df['raw_dialogue'].str.strip().str.replace('</i>|<i>', '', regex=True)
        
        # Split start and end times apart using separator and store in two separate columns
        subs_df[['raw_start_time', 'raw_end_time']] = subs_df['raw_time_str'].str.split(time_sep, n=1, expand=True)
        
        # Convert str -> timedelta
        subs_df['start_time'] = pd.to_timedelta(subs_df['raw_start_time'])
        subs_df['end_time'] = pd.to_timedelta(subs_df['raw_end_time'])
        
        movie_name = utils.remove_ext(os.path.basename(sub_file))
        movie_subs_dict[movie_name] = subs_df
        
    return movie_subs_dict


def extract_vsd_annotations(path):
    # os.path.join(vsd_annotations_dir, annot_filepath)
    with open(path, 'r') as fileobj:
        raw_lines = [x.strip() for x in fileobj.readlines() if x != '\n']
        
    df_rows = []
    for line in raw_lines:
        vals = line.split(' ', maxsplit=2)
        row = {'start': float(vals[0]), 'finish': float(vals[1])}
        
        if len(vals) > 2:
            row['desc'] = vals[2]
        df_rows.append(row)
        
    return pd.DataFrame(df_rows)