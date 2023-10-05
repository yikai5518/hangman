from arguments import get_args
from environment import HangmanEnv
from dataset import prepare_dataset
from policy import HangmanPolicy

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def main(args):
    train_words, eval_words, max_word_length = prepare_dataset("words_250000_train.txt")
    
    # Save words to save preprocessing time
    with open("train_words.txt", "w") as f:
        for word, orig in train_words:
            f.write(word + " " + orig + "\n")
    with open("eval_words.txt", 'w') as f:
        for word, orig in eval_words:
            f.write(word + " " + orig + "\n")
    
    train_envs = make_vec_env(
        HangmanEnv,
        n_envs=args.num_processes,
        seed=args.seed,
        env_kwargs=dict(
            words=train_words,
            max_word_length=max_word_length,
            num_lives=args.num_lives
        )
    )
        
    agent = PPO(
        policy=HangmanPolicy,
        env=train_envs,
        policy_kwargs=dict(
            output_dim_pi=64,
            output_dim_vf=64,
            hidden_dim=args.hidden_dim,
        ),
        verbose=1,
    )
    agent.learn(total_timesteps=args.num_steps)
    
    
if __name__ == "__main__":
    args = get_args()
    main(args)