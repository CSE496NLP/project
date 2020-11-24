import argparse
import os
import mmap

import pandas

import data
from data_preprocess import process_raw_line
import db
from ppdb import parse_ppdb

def file_iter(path):
    with open(path, 'rb') as f:
        map_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return iter(map_file.readline, b"")

def read_dataset(src_path, dst_path, vocab, pos_vocab):
    src_iter = file_iter(src_path)
    dst_iter = file_iter(dst_path)
    
    data = zip(src_iter, dst_iter)
    rows = map(lambda line: process_raw_line(line[0], line[1], vocab, pos_vocab), data)
    
    return pandas.DataFrame(rows)

def save_dataset(df, db_path):
    conn = db.create_connection(db_path)
    df.to_sql('lines', conn, chunksize=500, method='multi')
    conn.close()

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
        default='wiki.full.aner.ori')
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
    argparser.add_argument(
        '--ppdb-path',
        type=str,
        dest='ppdb_path'
    )
    args = argparser.parse_args()

    if args.output_dir == None:
        args.output_dir = os.path.join(WORK, 'editnts-ppdb-data', args.dataset)
    input_src_sent = os.path.join(args.data_path, args.dataset, f"{args.file_base}.{args.dataset_part}.src")
    input_dst_sent = os.path.join(args.data_path, args.dataset, f"{args.file_base}.{args.dataset_part}.dst")

    assert os.path.exists(args.vocab_path), "Vocab path should exist"

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

    print("Reading vocab")
    vocab = data.Vocab()
    vocab.add_vocab_from_file('./vocab_data/vocab.txt')
    pos_vocab = data.POSvocab('./vocab_data/')

    print("Reading dataset...")
    df = read_dataset(input_src_sent, input_dst_sent, vocab, pos_vocab)
    print("Saving dataset...")
    save_dataset(df, f"db/{args.file_base}.{args.dataset_part}")

    if(args.ppdb_path != None):
        print("Reading PPDB...")
        df = parse_ppdb(args.ppdb_path, vocab, pos_vocab)
        print("Saving PPDB...")
        save_dataset(df, "db/ppdb.db")
        
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
