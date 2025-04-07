# run.py
#!/usr/bin/env python3
import argparse
import os
import sys

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description='DocMind: Intelligent Document Management System')
    parser.add_argument('--mode', type=str, default='app', choices=['app', 'test', 'benchmark'],
                        help='Run mode: app (default), test, or benchmark')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing documents to process')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set data directory in environment variable
    os.environ['DATA_DIR'] = args.data_dir
    
    if args.mode == 'app':
        # Run the application
        from app.main import main as run_app
        run_app()
    elif args.mode == 'test':
        # Run the tests
        from app.tests import run_benchmark_tests
        run_benchmark_tests()
    elif args.mode == 'benchmark':
        # Run optimization
        from app.tests import optimize_system_parameters
        optimize_system_parameters()

if __name__ == "__main__":
    main()