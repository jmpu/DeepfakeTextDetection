from re import T
import jsonlines
import string
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from tqdm import tqdm
import nltk
from glob import glob
nltk.download('punkt')
import numpy as np

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cache_dir',
        default='/rdata/zainsarwar865/models',
        type=str,
        required=False,
        help="Directory where models are stored"
       )
    

    parser.add_argument(
        '--input_dir',
        default=None,
        type=str,
        required=True,
       )
    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        required=True,
        help="Output directory location"
       )
    
    parser.add_argument(
        '--grover_ds',
        default=False,
        type=int,
        required=True,
        help="Formatting requirement. Grover datasets are formatted differently. True if processing a Grover-styled dataset. False otherwise"
       )
    
    parser.add_argument(
        '--len_disc',
        default=False,
        type=int,
        required=True,
        help="Context window of the defense. If 1024, set to true. Else set false"
       )
    

    args = parser.parse_args()
    #all_tokenized_sentences = []
    #all_labels = []

    all_input_files = glob(args.input_dir)
    for file in all_input_files:
        output_file = args.output_dir + file.split('/')[-1]
        with jsonlines.open(file, 'r') as input_articles, jsonlines.open(output_file, 'w') as out_file:
            for idx, article in enumerate(input_articles):
                if(args.grover_ds):                    
                    article['text'] = article.pop('article')
                
                if(args.len_disc):
                    article['text'] = ' '.join(article['text'].split(' ')[0:1024])
                else:
                    article['text'] = ' '.join(article['text'].split(' ')[0:512])

                tokenized_sentences = sent_tokenize(article['text'])
                article['id'] = idx
                if(len(tokenized_sentences ) > 0):
                    if(tokenized_sentences[-1][-1] != "."):
                        tokenized_sentences = tokenized_sentences[0:-1]
                    
                    tot_sentences = len(tokenized_sentences)
                    first_div = None
                    second_div =None
                    third_div = None
                    fourth_div =None
                    fifth_div = None
                    sixth_div = None
                    seventh_div = None



                    if(args.len_disc == False):
                        if(tot_sentences > 18):
                            increment = tot_sentences // 4
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:2*increment]
                            third_div = tokenized_sentences[increment*2:3*increment]
                            fourth_div = tokenized_sentences[increment*3:]

                            # Three divisions
                        elif(tot_sentences <=18 and tot_sentences > 12):
                            # Two divisions
                            increment = tot_sentences // 3
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:2*increment]
                            third_div = tokenized_sentences[increment*2:]
                        
                        elif(tot_sentences <= 12 and tot_sentences > 6):
                            increment = tot_sentences // 2
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:2*increment]
                        else:
                            first_div = tokenized_sentences
                    
                    else:
                        # Make 7 partitions at max

                        if(tot_sentences > 36):
                            increment = tot_sentences // 7
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:2*increment]
                            third_div = tokenized_sentences[increment*2:3*increment]
                            fourth_div = tokenized_sentences[increment*3:increment*4]
                            fifth_div = tokenized_sentences[increment*4:increment*5]
                            sixth_div = tokenized_sentences[increment*5:increment*6]
                            seventh_div = tokenized_sentences[increment*6:]
                        
                        elif(tot_sentences <=36 and tot_sentences > 30):
                            increment = tot_sentences // 6
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:2*increment]
                            third_div = tokenized_sentences[increment*2:3*increment]
                            fourth_div = tokenized_sentences[increment*3:increment*4]
                            fifth_div = tokenized_sentences[increment*4:increment*5]
                            sixth_div = tokenized_sentences[increment*5:]
                                                    

                        
                        elif(tot_sentences <=30 and tot_sentences > 24):
                            increment = tot_sentences // 5
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:2*increment]
                            third_div = tokenized_sentences[increment*2:3*increment]
                            fourth_div = tokenized_sentences[increment*3:increment*4]
                            fifth_div = tokenized_sentences[increment*4:]

                        elif(tot_sentences <=24 and tot_sentences > 18):
                            increment = tot_sentences // 4
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:2*increment]
                            third_div = tokenized_sentences[increment*2:3*increment]
                            fourth_div = tokenized_sentences[increment*3:]

                            # Three divisions
                        elif(tot_sentences <=18 and tot_sentences > 12):
                            # Two divisions
                            increment = tot_sentences // 3
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:2*increment]
                            third_div = tokenized_sentences[increment*2:]

                        
                        elif(tot_sentences <= 12 and tot_sentences > 6):
                            increment = tot_sentences // 2
                            first_div = tokenized_sentences[0:increment]
                            second_div = tokenized_sentences[increment:]
                        else:
                            first_div = tokenized_sentences
                    
                    all_divisons_final = [first_div ,second_div,third_div ,fourth_div,fifth_div ,sixth_div,seventh_div]

                    for division in all_divisons_final:
                        if(division != None):
                            if(args.grover_ds):
                                article['article'] = ' '.join(division)
                                article.pop('text', None)
                            else:
                                article['text'] =  ' '.join(division)
                            out_file.write(article)
                else:
                    print("Empty file in file {}".format(file))                            



if __name__ == "__main__":
	main()
