#!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
import time

from tqdm import tqdm
from tensorboardX import SummaryWriter

## for training on server
import sys
sys.path.append('/efs_mint/Documents/LearningToPaint')
# exp = os.path.abspath('.').split('/')[-1]
# writer = TensorBoard('../train_log/{}'.format(exp))

# os.system('ln -sf ../train_log/{} ./log'.format(exp))
# os.system('mkdir ./model')

def train(agent, env, evaluate, writer):
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    resume = args.resume
    output = args.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = args.noise_factor
    pbar = tqdm(total = train_times+1)
    limit_act = args.limit_act; print(f"limit actions: {limit_act}")
    while step <= train_times:
        # print(f"step: {step}")
        step += 1
        pbar.update(1)
        episode_steps += 1
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)    
        action = agent.select_action(observation, noise_factor=noise_factor)
        def limit_actions(action, no_rbg=False, no_curve=False, no_transpar=False, same_thick=False):
            x = action.reshape(-1, 10 + 3)
            if no_rbg:
                x[:,-3:] = 1
            if no_curve:
                x[:,2:4] = 0
            if no_transpar:
                x[:,7] = 1
                x[:,9] = 1
            if same_thick:
                x[:,6] = x[:,8]

            x = x.reshape(-1, (10 + 3)*5)
            return x
            pass
        if limit_act: 
            action = limit_actions(action, no_rbg=True, no_curve=True, no_transpar=True, same_thick=True)

        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done, step)
        if (episode_steps >= max_step and max_step):
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, train_step=step, debug=debug)
                    if debug: prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(output)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            if step > args.warmup:
                if step < 10000 * max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr) # update gan
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
            if debug: prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                .format(episode, step, train_time_interval, time.time()-time_stamp)) 
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
    pbar.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint')

    # hyper-parameter
    parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95**5, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
    parser.add_argument('--train_times', default=2000000, type=int, help='total traintimes')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='output model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--limit_act', default=False, dest='limit_act', action='store_true', help='limit action space')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--identifier', default='test', type=str, help='experiment identifier')
    parser.add_argument('--data', default='./data/dump_cubi_test/', type=str, help='dataset path')

    args = parser.parse_args()    
    args.output = get_output_folder(args.output, args.identifier)
    

    from utils.utils import getWriterPath
    writer = SummaryWriter(getWriterPath(task='train',
                                              exper_name=args.identifier, date=True))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    from DRL.ddpg import DDPG
    from DRL.multi import fastenv
    fenv = fastenv(args.data, args.max_step, args.env_batch, writer)
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize, \
                 writer, args.resume, args.output)
    evaluate = Evaluator(args, writer)
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
    train(agent, fenv, evaluate, writer)
