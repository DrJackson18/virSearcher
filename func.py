#!/usr/bin/python3

import os


def run_proc(command_line):
    """
    Run a single command as if it were on the console

    :param command_line: the command line to be run
    """
    return_code = os.system(command_line)
    # Check whether it ran smoothly
    if return_code != 0:
        print("The command did not end well.\nExiting...")
        quit()


def remove_ext(file):
    """
    Remove the extension of a file path

    :param file: the file whose extension is to be removed
    """
    (filedir, tempfilename) = os.path.split(file)
    (filename, extension) = os.path.splitext(tempfilename)
    return os.path.join(filedir, filename)

