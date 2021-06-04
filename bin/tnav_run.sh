#!/bin/bash
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries/linux/mpi/intel64/lib
/opt/RFD/tNavigator-con-mpi --server-url $1 -r --ecl-rsm -n 32 $2