"""Run small tutorials through the public Python API, after package installation."""
from pathlib import Path
import argparse
import json
from priorlens import run
from priorlens.workflows import run_risk_summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=['nig','eight_schools','lupus','custom_1d','custom_2d'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--include-lin',action='store_true')
    args=parser.parse_args()
    if args.include_lin and args.case.startswith('custom'):
        parser.error('External-adapter Lin is not implemented')
    args.output.mkdir(parents=True,exist_ok=False)
    folder=Path(__file__).resolve().parent/args.case
    if args.case=='lupus':matching,risk='quick_matching.yaml','quick_risk.yaml'
    elif args.case.startswith('custom'):matching=risk='importance.yaml'
    else:matching=risk='quick.yaml'
    if args.include_lin:
        matching='matching_with_lin.yaml' if args.case=='lupus' else 'all_methods.yaml'
    reports=[run(folder/matching,args.output/'methods'),
             run_risk_summary(folder/risk,args.output/'risk_both')]
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
