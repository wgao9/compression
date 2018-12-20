#!/bin/bash
#!/usr/bin/python3
python3 deep_compress.py --type=mnist --mode=normal --fix_bit=2 --number_of_models=5 --result_root=results --result_name=exp3
python3 deep_compress.py --type=mnist --mode=hessian --fix_bit=2 --number_of_models=5 --diameter_reg=30 --result_root=results --result_name=exp4
python3 deep_compress.py --type=mnist --mode=normal --fix_bit=2 --number_of_models=5 --result_root=results --result_name=exp5
python3 deep_compress.py --type=mnist --mode=hessian --fix_bit=2 --number_of_models=5 --diameter_reg=30 --result_root=results --result_name=exp6
