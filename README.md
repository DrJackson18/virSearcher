# virSearcher
Identifying Bacteriophages from Metagenomes by Combining CNN and Gene Information
-----------
virSearcher which combines CNN and gene information is developed for identifing phages from metagenomic contigs.
## Reference
Identifying Bacteriophages from Metagenomes by Combining Convolutional Neural Network and Gene Information
## Requirement and Dependency
The system on which virSearcher is run should be Linux with Python3. The specific version of Python selected for virSearcher's development is Python3.6. The following Python packages should be installed:<br>
* numpy1.19.2<br>
* pandas0.20.3<br>
* biopython1.78<br>
* tensorflow1.14.0 or tensorflow-gpu1.14.0<br>
* keras2.3.0

If the computer is equipped with GPU, install the GPU version of tensorflow. Otherwise, select the CPU version. The version number of the packages should be as close to the above as possible to avoid some runtime errors. The suggested command for the installation is shown as

``pip3 install numpy==1.19.2 pandas==0.20.3 biopython==1.78 tensorflow-gpu==1.14.0 keras==2.3.0``

* BLAST+
BLAST+ should be downloaded(from https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download), installed properly and the environment variable should be configured so that virSearcher can call it through the simple command "blastp" without providing any directory.

## Setup

Set up virSearcher by:

``git clone https://github.com/liuql2019/virSearcher``

The gene databases should be downloaded from https://www.jianguoyun.com/p/DfXqt9YQ7I_kBxjSge8D. Decompress the database using setup_database.py provided by virSearcher:

``python3 setup_database.py -d database.zip``

## Usage

To predct metagenomic contigs as phages or not, run virSearcher as:

``python3 predict.py -i INPUT_SEQS [-t THR] [-s OUTPUT_FILE] [-c THREAD] [--G] [--B]``

* INPUT_SEQS
The input file in FASTA format, where the metagenomic contigs are stored.
* THR
The threshold used to determine whether each contig is phage or not. The default value of the threshold is 0.5.
* OUTPUT_FILE
The output path of the result file, which will be stored in the folder of the input file by default.
* THREAD
The maximum number of CPU cores that can be used by virSearcher. If omitted, the default value 4 will be chosen.

If you need to rerun virSearcher for some reasons, and the gene scan or BLAST step has been done before, these two steps can be skipped.
* --G
Skip the step of gene scan.
* --B
Skip the step of BLAST.

## Demo
The contig file in the folder demo is used to demenstrate how to run virSearcher. Change the current directory to the folder of virSearcher, and enter the following command in the console:

``python3 predict.py -i ./demo/contigs.fasta -s ./result/contigs.csv -c 8``

## Error Solution

If the following error message appears:

``./FragGeneScan1.31/FragGeneScan: Permission denied``

The executable permissions of the files in the folder FragGeneScan1.31 must be given. Change the current directory to the folder of virSearcher, enter the following command in the console:

``chmod -R 750 ./FragGeneScan1.31``

## Acknowledgement

FragGeneScan has been included in virSearcher. Thanks to the authors of FragGeneScan.

## Licence

Codes here may be modified and used for any purpose.This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

