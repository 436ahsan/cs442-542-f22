#!/bin/bash

# Get an interactive job
#   on 2 nodes, each with 8 processes
qsub -I -l nodes=2:ppn=8
