import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent to play Hangman")
    
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000000,
        help="Number of steps to train for"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to use for training"
    )
    parser.add_argument(
        "--num-lives",
        type=int,
        default=6,
        help="Number of lives for hangman game"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for MLP"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for training"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for logging"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
    )
    
    return parser.parse_args()