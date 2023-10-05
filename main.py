from arguments import get_args
from environment import HangmanEnv
from dataset import prepare_dataset, read_file
from policy import HangmanPolicy, HangmanFeaturesExtractor

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

def main(args):
    print("Preparing dataset...")
    
    train_words, eval_words, max_word_length = prepare_dataset("words_250000_train.txt")
    
    # Save words to save preprocessing time
    with open("train_words.txt", "w") as f:
        for word, orig in train_words:
            f.write(word + " " + orig + "\n")
    with open("eval_words.txt", 'w') as f:
        for word, orig in eval_words:
            f.write(word + " " + orig + "\n")
    
    # train_words = read_file('train_words.txt')
    # eval_words = read_file('eval_words.txt')
    # max_word_length = 40
    
    print(f"{len(train_words)} training words, {len(eval_words)} evaluation words in dataset")
    
    train_envs = make_vec_env(
        HangmanEnv,
        n_envs=args.num_processes,
        seed=args.seed,
        env_kwargs=dict(
            words=train_words,
            max_word_length=max_word_length,
            num_lives=args.num_lives,
        )
    )
    
    eval_envs = make_vec_env(
        HangmanEnv,
        n_envs=args.num_processes,
        seed=args.seed,
        env_kwargs=dict(
            words=eval_words,
            max_word_length=max_word_length,
            num_lives=args.num_lives,
        )
    )
        
    agent = PPO(
        policy=HangmanPolicy,
        env=train_envs,
        policy_kwargs=dict(
            output_dim_vf=64,
            hidden_dim=args.hidden_dim,
            features_extractor_class=HangmanFeaturesExtractor,
            features_extractor_kwargs=dict(
                embedding_size=64,
            )
        ),
        verbose=1,
        device='cuda',
        tensorboard_log=args.log_dir,
    )
    
    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=args.log_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    
    print(f"Training for {args.num_steps} steps...")
    agent.learn(total_timesteps=args.num_steps, callback=eval_callback)
    
    model.save('ppo_hangman')
    
    
if __name__ == "__main__":
    args = get_args()
    main(args)