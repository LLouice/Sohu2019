#!/usr/bin/env bash
set -eux

cd ../.
python -u data_raw_trnval.py &>> ../logs/data.log

python -u data_raw_test.py &>>  ../logs/data.log

python -u data_title_trnval.py --lite &>>  ../logs/data.log

python -u data_title_trnval.py &>>  ../logs/data.log

python -u data_title_test.py &>>  ../logs/data.log

ehco "over!"