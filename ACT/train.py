from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG # must import first

import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt

from training.utils import *
from training.dataset_preproc import RobotDatasetPreprocessor
# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
args = parser.parse_args()
task = args.task

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)

# device
device = os.environ['DEVICE']


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def train_bc(train_dataloader, val_dataloader, policy_config):
    # import wandb
    import wandb
    
    # initialize wandb
    wandb.init(
        project="act-training",
        config={
            "policy": policy_config['policy_class'],
            "hidden_dim": policy_config.get('hidden_dim', 512),
            "lr": policy_config.get('lr', 1e-5),
            "batch_size_train": train_cfg['batch_size_train'],
            "batch_size_val": train_cfg['batch_size_val'],
            "seed": train_cfg['seed']
        }
    )
    
    # load policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)

    # load optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    # create checkpoint dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in range(train_cfg['num_epochs']):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                
                # Save best model
                best_ckpt_path = os.path.join(checkpoint_dir, f"policy_best_seed_{train_cfg['seed']}.ckpt")
                torch.save(policy.state_dict(), best_ckpt_path)
                wandb.save(best_ckpt_path)
                
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        
        # Log validation metrics to wandb
        val_metrics = {"epoch": epoch}
        for k, v in epoch_summary.items():
            val_value = v.item()
            summary_string += f'{k}: {val_value:.3f} '
            val_metrics[f"val/{k}"] = val_value
        print(summary_string)
        wandb.log(val_metrics)

        # training
        policy.train()
        optimizer.zero_grad()
        batch_losses = []
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            batch_losses.append(detach_dict(forward_dict))
            
            # Log batch metrics periodically
            if batch_idx % 10 == 0:
                batch_metrics = {"epoch": epoch, "batch": batch_idx}
                for k, v in detach_dict(forward_dict).items():
                    batch_metrics[f"batch/{k}"] = v.item()
                wandb.log(batch_metrics)
                
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        
        # Log training metrics to wandb
        train_metrics = {"epoch": epoch}
        for k, v in epoch_summary.items():
            train_value = v.item()
            summary_string += f'{k}: {train_value:.3f} '
            train_metrics[f"train/{k}"] = train_value
        print(summary_string)
        wandb.log(train_metrics)
        
        # Log learning rate
        wandb.log({"epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']})

        if epoch % 200 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])
            
            # Save plot to wandb
            plot_path = os.path.join(checkpoint_dir, f"loss_plot_seed_{train_cfg['seed']}.png")
            if os.path.exists(plot_path):
                wandb.log({"loss_plot": wandb.Image(plot_path)})

    # Save final model
    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    wandb.save(ckpt_path)
    
    # Close wandb run
    wandb.finish()
    

if __name__ == '__main__':
    # set seed
    set_seed(train_cfg['seed'])
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
   # number of training episodes
    task = 'pick_screwdriver/screwdriver'
    data_dir = os.path.join(task_cfg['dataset_dir'], task)
    num_episodes = len(os.listdir(data_dir))

    # preprocess dataset
    #preproc = RobotDatasetPreprocessor(dataset_dir=data_dir, target_length=task_cfg['episode_len'])
    #output_dir = preproc.preprocess_dataset()

    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(data_dir, num_episodes, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # train
    train_bc(train_dataloader, val_dataloader, policy_config)