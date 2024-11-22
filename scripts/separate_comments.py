import pandas as pd
import os
import re

input_folder = '../data/MADE-WIC/OSPR'
output_folder = '../data/MADE-WIC-c/OSPR'

def extract_comments(row):
    comment_regex = re.compile('((\/\*([\s\S]*?)\*\/)|((?<!:)\/\/.*))', re.MULTILINE)

    detected_comments = re.findall(comment_regex, row.Function)
    comments = ' '.join([comment[0] for comment in detected_comments])
    comments = row.LeadingComment + ' ' + comments

    only_code = re.sub(comment_regex, ' ', row.Function)

    return comments, only_code


if __name__ == "__main__":

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            df = pd.read_csv('{}/{}'.format(input_folder, filename))
            df.fillna('', inplace=True)

            df[['Comments', 'OnlyCode']] = df.apply(extract_comments, axis=1, result_type='expand')

            df.to_csv('{}/{}'.format(output_folder, filename))