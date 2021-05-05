#!/usr/bin/python3

import os
import func as f
import argparse

PARSER = argparse.ArgumentParser(description='Unzip the virus gene database.')
PARSER.add_argument("-d", type=str, action="store", dest="zip_data",
                    help="The zipped file of the gene database.", required=True)
args = PARSER.parse_args()

virSearcher_folder, _ = os.path.split(os.path.realpath(__file__))
f.run_proc("unzip "+os.path.realpath(args.zip_data)+" -d "+virSearcher_folder)
