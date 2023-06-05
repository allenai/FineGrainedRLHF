import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument(
        '--mode', type=str, choices=['train', 'eval'], required=True, help='train or eval?')

    # dataset
    parser.add_argument(
        '--data_path', type=str, default='../data/{datapath}/{split}.tsv')
    # '../data/knowledge/generated_gkp_gpt3curie.{split}.{task}.json'
    parser.add_argument(
        '--train_tasks', type=str, default='obqa')
    parser.add_argument(
        '--eval_tasks', type=str, default='obqa')
    parser.add_argument(
        '--eval_split', type=str, default='dev', choices=['dev', 'test'])

    # model
    parser.add_argument(
        '--run_model_ckpt', type=str, default="", help='trained policy')
    parser.add_argument(
        '--model_type', type=str, default='t5-large', help='model used for policy, ref policy, and value')
    parser.add_argument(
        '--model_ckpt', type=str, default='/gscratch/tial/yushihu/human_feedback/t5-large-oracle-1k-train', help='model ckpt used for policy and ref policy (NOT value!)')
    parser.add_argument(
        '--value_model_ckpt', type=str, default='t5-base', help='model ckpt used for policy and ref policy (NOT value!)')
    parser.add_argument(
        '--freeze_value_model', action='store_true', default=False)
    parser.add_argument(
        '--use_model_ckpt_for_value', action='store_true', default=False)
    parser.add_argument(
        '--policy_value_sharing', action='store_true', default=False)
    parser.add_argument(
        '--input_padding_side', type=str, default="left", help='padding side input prompt')  # right for Ellen's current version
    parser.add_argument(
        '--max_input_len', type=int, default=1024, help='max length of the input prompt')
    parser.add_argument(
        '--max_generated_len', type=int, default=200, help='max length of the output knowledge')
    parser.add_argument(
        '--load_from_ckpt', type=str, default=None, help='ckpt path to resume training or run eval')
    parser.add_argument(
        '--load_from_stageI_ckpt', type=str, default=None, help='ckpt path to resume training or run eval')

    # reward
    parser.add_argument(
        '--baseline_model_ckpt', type=str, default="/gscratch/tial/yushihu/human_feedback/baseline_rm_900_train_ellen0503", help='reward model checkpoint')
    parser.add_argument(
        '--non_factual_model_ckpt', type=str, default="/gscratch/tial/yushihu/human_feedback/non_fact_rm_ellen0425", help='reward model checkpoint')
    parser.add_argument(
        '--factual_model_ckpt', type=str, default="/gscratch/tial/yushihu/human_feedback/fact_rm_ellen0425", help='reward model checkpoint')
    parser.add_argument(
        '--completeness_model_ckpt', type=str, default="/gscratch/tial/yushihu/human_feedback/missing_rm_pref_coverage_ellen0503", help='reward model checkpoint')
    parser.add_argument(
        '--kl_coef', type=float, default=0.2, help='coefficient for KL term in reward')
    parser.add_argument(
        '--fine_grained', action='store_true', default=False, help='whether to use fine-grained reward')
    parser.add_argument(
        '--baseline_reward_mean', type=float, default=0.0, help='reward model mean')
    parser.add_argument(
        '--baseline_reward_std', type=float, default=1.0, help='reward model stdev')
    parser.add_argument(
        '--baseline_reward_bias', type=float, default=0.0, help='reward model mean')
    parser.add_argument(
        '--baseline_reward_scale', type=float, default=1.0, help='reward model stdev')
    parser.add_argument(
        '--verbosity_positive_reward', type=float, default=1.0, help='reward for a good sentence')
    parser.add_argument(
        '--verbosity_negative_reward', type=float, default=-1.0, help='reward for a bad sentence')
    parser.add_argument(
        '--factuality_positive_reward', type=float, default=1.0, help='reward for a good sentence')
    parser.add_argument(
        '--factuality_negative_reward', type=float, default=-1.0, help='reward for a bad sentence')
    parser.add_argument(
        '--completeness_reward_mean', type=float, default=0.0, help='reward model mean')
    parser.add_argument(
        '--completeness_reward_std', type=float, default=1.0, help='reward model stdev')
    parser.add_argument(
        '--completeness_reward_bias', type=float, default=0.0, help='reward model mean')
    parser.add_argument(
        '--completeness_reward_scale', type=float, default=1.0, help='reward model stdev')


    # ppo
    parser.add_argument(
        '--pg_coef', type=float, default=1.0, help='policy loss coefficient')
    parser.add_argument(
        '--vf_coef', type=float, default=1.0, help='value loss coefficient')
    parser.add_argument(
        '--cliprange', type=float, default=.2, help='clip parameter for policy gradient')
    parser.add_argument(
        '--cliprange_value', type=float, default=.2, help='clip parameter for value function')
    parser.add_argument(
        '--gamma', type=float, default=1.0, help='discount factor for rewards')
    parser.add_argument(
        '--lam', type=float, default=0.95, help='lambda parameter for generalized advantage estimation')
    parser.add_argument(
        '--whiten_rewards', action='store_false', default=True, help='whether to normalize reward in each minibatch')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')

    # train
    parser.add_argument(
        '--total_episodes', type=int, default=100000, help='total number of episodes')
    parser.add_argument(
        '--num_warmup_step_ratio', type=float, default=0.0, help = 'ratio of number of steps to use for warmup with linear warmup')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for sampling')
    parser.add_argument(
        '--training_batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument(
        '--noptepochs', type=int, default=4, help='number of ppo epochs reusing rollouts')
    parser.add_argument(
        '--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument(
        '--temperature', type=float, default=0.7, help='temperature for sampling from policy during training')
    parser.add_argument(
        '--num_samples', type=int, default=1, help='number of samples to collect for each episode')
    parser.add_argument(
        '--top_k', type=int, default=None, help='hyperparameter for top-k sampling')
    parser.add_argument(
        '--top_p', type=float, default=None, help='hyperparameter for nucleus sampling')
    parser.add_argument(
        '--kl_threshold', type=float, default=10.0, help='if training batch KL is more than this, stop training')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log_interval', type=int, default=1, help='step interval to print out logs')
    parser.add_argument(
        '--save_interval', type=int, default=500, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--nolog', default=False, action='store_true')
    parser.add_argument('--eval_loop_cap', type=int, default=None, help='cap on number of eval loops')

    parser.add_argument(
        '--eval_baseline', action='store_true', help='whether to evaluate the no-knowledge baseline')
    parser.add_argument(
        '--cuda_deterministic', action='store_false', default=True,
        help='sets flags for determinism when using CUDA (potentially slow!)')
    args = parser.parse_args()

    return args

