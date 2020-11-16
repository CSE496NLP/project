import argparse
import os

import data_preprocess
import db
from ppdb import parse_ppdb

def main():
    
    WORK = os.environ['WORK']
    COMMON_DATADIR='/common/cse896/asturtz/data-simplification/'
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--data-path',
         type=str,
         dest='data_path',
         default=COMMON_DATADIR
    )
    argparser.add_argument(
        '--dataset',
        type=str,
        dest='dataset',
        default='wikilarge'
    )
    argparser.add_argument(
        '--file-base',
        type=str,
        dest='file_base',
        default='wiki.full.aner.')
    argparser.add_argument(
        '--ppdb-version',
        type=str,
        dest='ppdb_version',
        default='ppdb-tldr')
    argparser.add_argument(
        '--dataset-part',
        type=str,
        dest='dataset_part',
        default='train'
    )
    argparser.add_argument(
        '--output-directory',
        type=str,
        dest='output_dir'
    )
    argparser.add_argument(
        '--vocab-path',
        type=str,
        dest='vocab_path',
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab_data') +'/'
    )
    args = argparser.parse_args()

    if args.output_dir == None:
        args.output_dir = os.path.join(WORK, 'editnts-ppdb-data', args.dataset)
    ppdb_path = os.path.join(args.data_path, 'ppdb', args.ppdb_version)
    input_src_sent = os.path.join(args.data_path, args.dataset, f"{args.file_base}{args.dataset_part}.src")
    input_dst_sent = os.path.join(args.data_path, args.dataset, f"{args.file_base}{args.dataset_part}.dst")

    assert os.path.exists(ppdb_path), "Ppdb path should exist"
    assert os.path.exists(args.vocab_path), "Vocab path should exist"
    assert os.path.exists('db/ppdb.db'), "PPDB database should exist"

    print(f"Arguments:")
    print(f"  Dataset            : {args.dataset}")
    print(f"  Partition          : {args.dataset_part}")
    print(f"  File Basename      : {args.file_base}")
    print(f"  PPDB Version:      : {args.ppdb_version}")
    print(f"Computed:")
    print(f"  Output Directory   : {args.output_dir}")
    print(f"  PPDB Path          : {ppdb_path}")
    print(f"  Input SRC Sentences: {input_src_sent}")
    print(f"  Input DST Sentences: {input_dst_sent}")

    print("Processing PPDB...")
    conn = db.create_connection('db/ppdb.db')
    parse_ppdb(ppdb_path, conn)
    conn.close()
        
    print("Done.")

    #print("Opening Source Sentences... ", end='')
    #with open(input_src_sent, 'r') as f:
    #    source_sentences = f.readlines()
    #print("Done.")

    #print("Opening Destination Sentences... ", end='')
    #with open(input_dst_sent, 'r') as f:
    #    dest_sentences = f.readlines()
    #print("Done.")

    #print("Processing raw data... ", end='')
    #data_frame = process_raw_data(source_sentences, dest_sentences, args.vocab_path)
    #print("Saving data frame...", end='') 
    #data_frame.to_pickle(os.path.join(args.output_dir, f'{args.dataset_part}.filtered.pos.'))
    #print("Done.")

    #print("Converting to IDs... ", end='')
    #editnet_data_to_editnetID(data_frame, os.path.join(args.output_dir, f'{args.dataset_part}.filtered.pos.'))
    #print("Done.")
    


if __name__ == '__main__':
    main()
