These scripts do not run HLS three time, every HLS saves reports in impl_report and need to be manually copied to report1
and report2. Together with impl_report they consists three HLS results.

Evey iteration takes approximatly 3-4h depending on hardware.

Inside every directory, run

$ python script.py

to run full HLS and report final result in graph

or run

$ python -O script.py

to regenerate result in graph based on previous HLS reports in impl_result, result1 and result2 without HLS.

Dependencies:

matplotlib:
Installation, For Python 3, this is:

$ sudo apt-get install python3-matplotlib

If you're using Python 2.7, this is:

$ sudo apt-get install python-matplotlib
