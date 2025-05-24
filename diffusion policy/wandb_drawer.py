import wandb
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import json
import os
from matplotlib.ticker import FuncFormatter


json_files = ["/home/daiying/diffusion_policy_with_kovis/diffusion_policy/data.json", "/home/daiying/diffusion_policy_with_kovis/diffusion_policy/data_object.json", "/home/daiying/diffusion_policy_with_kovis/diffusion_policy/data_spatial.json"]

# Configuration for first project
wandb_project1 = "openvla_finetune_screw"
wandb_project2 = "diffusion_policy_debug"  
wandb_project3 = "act-training"
wandb_project4 = "openpi"

wandb_entity1 = "hritdy"
wandb_entity2 = "hriday"

# metric_name1 = "action_accuracy"
# metric_name2 = "train_action_mse_error"
# metric_name3 = "train/l1"
metric_name1 = "train_loss"
metric_name2 = "train_loss"
metric_name3 = "train/loss"
metric_name4 = "loss"
total_epochs = 300000  # Total number of epochs

# Set font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

# Configure font sizes
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)   # legend fontsize

def load_and_process_json(filepath, batch_size=256):
    """
    Load a JSON file and extract the loss values with proper step scaling
    
    Args:
        filepath: Path to the JSON file
        batch_size: Batch size used in training (for scaling steps)
        
    Returns:
        tuple: (scaled_steps, loss_values)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract steps and loss (first value in each list)
    steps = [int(step) for step in data.keys()]
    loss_values = [data[str(step)][0] for step in steps]
    
    # Scale steps by batch size
    scaled_steps = [step * batch_size for step in steps]
    
    return scaled_steps, loss_values

def interpolate_metrics(steps, values, target_points=500):
    """
    Interpolate metrics to match target number of points
    
    Args:
        steps: Original step values
        values: Original data values
        target_points: Number of points for interpolation
        
    Returns:
        tuple: (interpolated_steps, interpolated_values)
    """
    if len(steps) <= 1:
        return steps, values
    
    # Create a range of evenly spaced steps
    min_step, max_step = min(steps), max(steps)
    interp_steps = np.linspace(min_step, max_step, target_points)
    
    # Create the interpolation function and apply it
    interp_func = interp1d(steps, values, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_values = interp_func(interp_steps)
    
    return interp_steps, interp_values

def process_multiple_files(json_files, batch_size=256, target_points=500, smooth_factor=0.6):
    """
    Process multiple JSON files and prepare them for plotting
    
    Args:
        json_files: List of JSON file paths
        batch_size: Batch size used for scaling steps
        target_points: Number of points for interpolation
        smooth_factor: Amount of smoothing to apply (0-1)
        
    Returns:
        dict: Dictionary containing processed data for each file
    """
    results = {}
    max_steps = 0
    
    # Process each file
    for filepath in json_files:
        file_name = os.path.basename(filepath).replace('.json', '')
        steps, loss_values = load_and_process_json(filepath, batch_size)
        
        # Skip if no data
        if not steps or not loss_values:
            print(f"No valid data found in {file_name}")
            continue
        
        # Update max steps for uniform scaling later
        max_steps = max(max_steps, max(steps))
            
        # Interpolate to get smooth, uniform data
        interp_steps, interp_loss = interpolate_metrics(steps, loss_values, target_points)
        
        # Apply exponential smoothing if requested
        if smooth_factor > 0:
            alpha = 1 - smooth_factor
            smoothed_loss = np.array(interp_loss)
            for j in range(1, len(smoothed_loss)):
                smoothed_loss[j] = alpha * interp_loss[j] + (1 - alpha) * smoothed_loss[j-1]
            interp_loss = smoothed_loss
        
        # Store data
        results[file_name] = {
            'raw_steps': steps,
            'raw_loss': loss_values,
            'interp_steps': interp_steps,
            'interp_loss': interp_loss
        }
    
    # Create common x-axis steps for all files
    common_steps = np.linspace(0, max_steps, target_points)
    
    # Re-interpolate all data to use the common steps
    for file_name in results:
        steps = results[file_name]['raw_steps']
        loss = results[file_name]['raw_loss']
        
        if len(steps) > 1:
            interp_func = interp1d(steps, loss, kind='linear', bounds_error=False, fill_value='extrapolate')
            interp_loss = interp_func(common_steps)
            
            # Apply smoothing again if needed
            if smooth_factor > 0:
                alpha = 1 - smooth_factor
                smoothed_loss = np.array(interp_loss)
                for j in range(1, len(smoothed_loss)):
                    smoothed_loss[j] = alpha * interp_loss[j] + (1 - alpha) * smoothed_loss[j-1]
                interp_loss = smoothed_loss
                
            results[file_name]['common_steps'] = common_steps
            results[file_name]['common_loss'] = interp_loss
    
    return results, common_steps


# Initialize wandb API
api = wandb.Api()

def get_project_metrics(wandb_entity, project_name, record_runs, metric_name, scale, invert=False):
    runs = api.runs(f"{wandb_entity}/{project_name}")
    all_metrics = []
    run_names = []
    
    for run in runs:
        if run.name not in record_runs:
            continue
        if metric_name in run.history(keys=[metric_name]):
            history = run.history(keys=[metric_name])
            metric_values = history[metric_name].dropna().values
            metric_values = np.clip(metric_values, 0, 1)
            if invert:
                accuracy = 1 - metric_values / scale
            # else:
            #     accuracy = metric_values
                metric_values = np.clip(accuracy, 0, 1)  # Ensure accuracy is in [0,1]
            all_metrics.append(metric_values)
            run_names.append(run.name)
    
    # Align metrics
    max_length = min(map(len, all_metrics))
    aligned_metrics = [metric[:max_length] for metric in all_metrics]
    aligned_metrics = np.array(aligned_metrics)
    
    return aligned_metrics, run_names, max_length

def interpolate_metrics(metrics, original_points, target_points):
    """Interpolate metrics to match target number of points"""
    original_x = np.linspace(0, total_epochs, original_points)
    target_x = np.linspace(0, total_epochs, target_points)
    
    interpolated_metrics = []
    for metric in metrics:
        interpolator = interp1d(original_x, metric, kind='linear')
        interpolated_metrics.append(interpolator(target_x))
    
    return np.array(interpolated_metrics)

# Get metrics for both projects
record_runs1 = ['run1', 'run2', 'run3']
record_runs2 = ['run1', 'run2', 'run3']
record_runs3 = ['run1', 'run2', 'run3'] 
record_runs4 = ['run1', 'run2', 'run3'] 

# aligned_metrics1, run_names1, max_length1 = get_project_metrics(wandb_entity1, wandb_project1, record_runs1, metric_name1, 1,  invert=False)
# aligned_metrics2, run_names2, max_length2 = get_project_metrics(wandb_entity2, wandb_project2, record_runs2, metric_name2, 2.25, invert=True)
# aligned_metrics3, run_names3, max_length3 = get_project_metrics(wandb_entity2, wandb_project3, record_runs3, metric_name3, 1, invert=True)

aligned_metrics1, run_names1, max_length1 = get_project_metrics(wandb_entity1, wandb_project1, record_runs1, metric_name1, 1, invert=False)
aligned_metrics2, run_names2, max_length2 = get_project_metrics(wandb_entity2, wandb_project2, record_runs2, metric_name2, 1, invert=False)
aligned_metrics3, run_names3, max_length3 = get_project_metrics(wandb_entity2, wandb_project3, record_runs3, metric_name3, 1, invert=False)
aligned_metrics4, run_names4, max_length4 = get_project_metrics(wandb_entity2, wandb_project4, record_runs4, metric_name4, 1, invert=False)
# Create common x-axis points (500 points)
target_points = 500

# Interpolate metrics for project 2 to match project 1's points
if max_length2 != target_points:
    aligned_metrics2 = interpolate_metrics(aligned_metrics2, max_length2, target_points)

# Ensure project 1 also has the target number of points
if max_length1 != target_points:
    aligned_metrics1 = interpolate_metrics(aligned_metrics1, max_length1, target_points)

if max_length3 != target_points:
    aligned_metrics3 = interpolate_metrics(aligned_metrics3, max_length3, target_points)

if max_length4 != target_points:
    aligned_metrics4 = interpolate_metrics(aligned_metrics4, max_length4, target_points)

# Compute mean and variance for both projects
mean_metrics1 = np.mean(aligned_metrics1, axis=0)
std_metrics1 = np.std(aligned_metrics1, axis=0)

mean_metrics2 = np.mean(aligned_metrics2, axis=0)
std_metrics2 = np.std(aligned_metrics2, axis=0)

mean_metrics3 = np.mean(aligned_metrics3, axis=0)
std_metrics3 = np.std(aligned_metrics3, axis=0)

mean_metrics4 = np.mean(aligned_metrics4, axis=0)
std_metrics4 = np.std(aligned_metrics4, axis=0)
# Create x-axis points that represent epochs
x_points = np.linspace(0, total_epochs, target_points)

# Plotting
plt.figure(figsize=(10, 6))

# Plot first project (blue)
plt.plot(x_points, mean_metrics1, label=f"OpenVLA", color="blue", linewidth=1.5)
#upper_bound1 = np.clip(mean_metrics1 + std_metrics1, 0, 1)
upper_bound1 = mean_metrics1 + std_metrics1
#lower_bound1 = np.clip(mean_metrics1 - std_metrics1, 0, 1)
lower_bound1 = mean_metrics1 - std_metrics1
plt.fill_between(x_points, lower_bound1, upper_bound1, color="blue", alpha=0.2)

# plt.fill_between(x_points, 
#                  mean_metrics1 - std_metrics1, 
#                  mean_metrics1 + std_metrics1, 
#                  color="blue", alpha=0.2, 
#                  label=f"Variance (OpenVLA)")

# Plot second project (red)
plt.plot(x_points, mean_metrics2, label=f"Diffusion Policy", color="red", linewidth=1.5)
# upper_bound2 = np.clip(mean_metrics2 + std_metrics2, 0, 1)
# lower_bound2 = np.clip(mean_metrics2 - std_metrics2, 0, 1)
upper_bound2 = mean_metrics2 + std_metrics2
lower_bound2 = mean_metrics2 - std_metrics2
plt.fill_between(x_points, lower_bound2, upper_bound2, color="red", alpha=0.2)

# Plot third project (green)
plt.plot(x_points, mean_metrics3, label=f"ACT", color="green", linewidth=1.5)
# upper_bound2 = np.clip(mean_metrics2 + std_metrics2, 0, 1)
# lower_bound2 = np.clip(mean_metrics2 - std_metrics2, 0, 1)
upper_bound3 = mean_metrics3 + std_metrics3
lower_bound3 = mean_metrics3 - std_metrics3
plt.fill_between(x_points, lower_bound3, upper_bound3, color="green", alpha=0.2)


# Plot forth project (gray)
plt.plot(x_points, mean_metrics4, label=r"$\pi_0$", color="gray", linewidth=1.5)
# upper_bound2 = np.clip(mean_metrics2 + std_metrics2, 0, 1)
# lower_bound2 = np.clip(mean_metrics2 - std_metrics2, 0, 1)
upper_bound4 = mean_metrics4 + std_metrics4
lower_bound4 = mean_metrics4 - std_metrics4
plt.fill_between(x_points, lower_bound4, upper_bound4, color="gray", alpha=0.2)

plt.title("")
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Train Loss", fontsize=20)
# Configure legend
plt.legend(
    loc='upper right',  # Position of the legend: 'best', 'upper right', 'lower left', etc.
    prop={'size': 20, 'family': 'Arial'},
    #bbox_to_anchor=(1, 0.5),  # Place legend outside the plot
    frameon=True,  # Add a frame to the legend
    shadow=True,  # Add shadow to the frame
    fancybox=True  # Round corners
)

# Add grid and adjust layout
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent label clipping

# Show or save the plot
plt.show()
print('a')

# import wandb
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import interp1d
# import json
# import os
# from matplotlib.ticker import FuncFormatter


# # List of JSON files to process (NORA data)
# json_files = [
#     "/home/daiying/diffusion_policy_with_kovis/diffusion_policy/data.json", 
#     "/home/daiying/diffusion_policy_with_kovis/diffusion_policy/data_object.json", 
#     "/home/daiying/diffusion_policy_with_kovis/diffusion_policy/data_spatial.json"
# ]

# # Configuration for wandb projects
# wandb_project1 = "openvla_finetune_screw"
# wandb_project2 = "diffusion_policy_debug"  
# wandb_project3 = "act-training"
# wandb_project4 = "openpi"

# wandb_entity1 = "hritdy"
# wandb_entity2 = "hriday"

# # Metric names for different projects
# metric_name1 = "train_loss"
# metric_name2 = "train_loss"
# metric_name3 = "train/loss"
# metric_name4 = "loss"
# total_epochs = 300000  # Total number of epochs

# # Set font sizes
# SMALL_SIZE = 12
# MEDIUM_SIZE = 14
# BIGGER_SIZE = 20

# # Configure font sizes
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

# # Run IDs for each wandb project
# record_runs1 = ['run1', 'run2', 'run3']
# record_runs2 = ['run1', 'run2', 'run3']
# record_runs3 = ['run1', 'run2', 'run3'] 
# record_runs4 = ['run1', 'run2', 'run3'] 

# def load_and_process_json(filepath):
#     """
#     Load a JSON file and extract the loss values without applying batch size scaling yet
    
#     Args:
#         filepath: Path to the JSON file
        
#     Returns:
#         tuple: (steps, loss_values)
#     """
#     with open(filepath, 'r') as f:
#         data = json.load(f)
    
#     # Extract steps and loss (first value in each list)
#     steps = [int(step) for step in data.keys()]
#     loss_values = [data[str(step)][0] for step in steps]
    
#     # Clip loss values to 1.0
#     loss_values = np.clip(loss_values, 0, 1)
    
#     return steps, loss_values

# def interpolate_metrics(steps, values, target_points=500):
#     """
#     Interpolate metrics to match target number of points
    
#     Args:
#         steps: Original step values
#         values: Original data values
#         target_points: Number of points for interpolation
        
#     Returns:
#         tuple: (interpolated_steps, interpolated_values)
#     """
#     if len(steps) <= 1:
#         return steps, values
    
#     # Create a range of evenly spaced steps
#     min_step, max_step = min(steps), max(steps)
#     interp_steps = np.linspace(min_step, max_step, target_points)
    
#     # Create the interpolation function and apply it
#     interp_func = interp1d(steps, values, kind='linear', bounds_error=False, fill_value='extrapolate')
#     interp_values = interp_func(interp_steps)
    
#     return interp_steps, interp_values

# def process_multiple_files(json_files, nora_batch_size=256, nora_scaling_factor=1.0, target_points=500, smooth_factor=0.6):
#     """
#     Process multiple JSON files and prepare them for plotting
    
#     Args:
#         json_files: List of JSON file paths
#         nora_batch_size: Batch size used by NORA for scaling steps
#         nora_scaling_factor: Additional scaling factor to adjust NORA curve alignment (default: 1.0)
#         target_points: Number of points for interpolation
#         smooth_factor: Amount of smoothing to apply (0-1)
        
#     Returns:
#         tuple: (results dict, common_steps, mean_loss, std_loss)
#     """
#     results = {}
#     max_steps = 0
#     all_loss_data = []
    
#     # Process each file
#     for filepath in json_files:
#         file_name = os.path.basename(filepath).replace('.json', '')
#         steps, loss_values = load_and_process_json(filepath)
        
#         # Clip loss values to 1.0
#         loss_values = np.clip(loss_values, 0, 1)
        
#         # IMPORTANT CHANGE: Apply the batch size scaling to steps (multiply by batch size)
#         # This stretches the NORA data to account for batch size difference
#         # Additionally apply the scaling factor to fine-tune the curve placement
#         scaled_steps = [step * nora_batch_size * nora_scaling_factor for step in steps]
        
#         # Update max steps for uniform scaling later
#         if scaled_steps:
#             max_steps = max(max_steps, max(scaled_steps))
#             # Ensure max_steps doesn't exceed total_epochs
#             max_steps = min(max_steps, total_epochs)
            
#         # Interpolate to get smooth, uniform data
#         interp_steps, interp_loss = interpolate_metrics(scaled_steps, loss_values, target_points)
        
#         # Apply exponential smoothing if requested
#         if smooth_factor > 0:
#             alpha = 1 - smooth_factor
#             smoothed_loss = np.array(interp_loss)
#             for j in range(1, len(smoothed_loss)):
#                 smoothed_loss[j] = alpha * interp_loss[j] + (1 - alpha) * smoothed_loss[j-1]
#             interp_loss = smoothed_loss
        
#         # Store data
#         results[file_name] = {
#             'raw_steps': scaled_steps,  # Now using scaled steps
#             'raw_loss': loss_values,
#             'interp_steps': interp_steps,
#             'interp_loss': interp_loss
#         }
    
#     # Create common x-axis steps for all files
#     common_steps = np.linspace(0, max_steps, target_points)
    
#     # Re-interpolate all data to use the common steps
#     for file_name in results:
#         steps = results[file_name]['raw_steps']
#         loss = results[file_name]['raw_loss']
        
#         if len(steps) > 1:
#             interp_func = interp1d(steps, loss, kind='linear', bounds_error=False, fill_value='extrapolate')
#             interp_loss = interp_func(common_steps)
            
#             # Apply smoothing again if needed
#             if smooth_factor > 0:
#                 alpha = 1 - smooth_factor
#                 smoothed_loss = np.array(interp_loss)
#                 for j in range(1, len(smoothed_loss)):
#                     smoothed_loss[j] = alpha * interp_loss[j] + (1 - alpha) * smoothed_loss[j-1]
#                 interp_loss = smoothed_loss
            
#             # Clip interpolated loss values to 1.0 again
#             interp_loss = np.clip(interp_loss, 0, 1)
                
#             results[file_name]['common_steps'] = common_steps
#             results[file_name]['common_loss'] = interp_loss
            
#             # Add to collection for computing mean and variance
#             all_loss_data.append(interp_loss)
    
#     # Compute mean and standard deviation across all JSON files
#     if all_loss_data:
#         all_loss_array = np.array(all_loss_data)
#         mean_loss = np.mean(all_loss_array, axis=0)
#         std_loss = np.std(all_loss_array, axis=0)
#     else:
#         mean_loss = np.array([])
#         std_loss = np.array([])
    
#     return results, common_steps, mean_loss, std_loss

# def get_project_metrics(wandb_entity, project_name, record_runs, metric_name, scale, invert=False):
#     """
#     Get metrics from wandb project
    
#     Args:
#         wandb_entity: Wandb entity name
#         project_name: Wandb project name
#         record_runs: List of run names to include
#         metric_name: Metric name to extract
#         scale: Scale factor for the metric
#         invert: Whether to invert the metric
        
#     Returns:
#         tuple: (aligned_metrics, run_names, max_length)
#     """
#     api = wandb.Api()
#     runs = api.runs(f"{wandb_entity}/{project_name}")
#     all_metrics = []
#     run_names = []
    
#     for run in runs:
#         if run.name not in record_runs:
#             continue
#         if metric_name in run.history(keys=[metric_name]):
#             history = run.history(keys=[metric_name])
#             metric_values = history[metric_name].dropna().values
#             metric_values = np.clip(metric_values, 0, 1)
#             if invert:
#                 accuracy = 1 - metric_values / scale
#                 metric_values = np.clip(accuracy, 0, 1)  # Ensure accuracy is in [0,1]
#             all_metrics.append(metric_values)
#             run_names.append(run.name)
    
#     # Align metrics
#     if all_metrics:
#         max_length = min(map(len, all_metrics))
#         aligned_metrics = [metric[:max_length] for metric in all_metrics]
#         aligned_metrics = np.array(aligned_metrics)
#         return aligned_metrics, run_names, max_length
#     else:
#         return np.array([]), [], 0

# def interpolate_metrics_to_target(metrics, original_points, target_points):
#     """
#     Interpolate metrics to match target number of points
    
#     Args:
#         metrics: Original metrics
#         original_points: Number of original points
#         target_points: Number of target points
        
#     Returns:
#         numpy.ndarray: Interpolated metrics
#     """
#     original_x = np.linspace(0, total_epochs, original_points)
#     target_x = np.linspace(0, total_epochs, target_points)
    
#     interpolated_metrics = []
#     for metric in metrics:
#         interpolator = interp1d(original_x, metric, kind='linear')
#         interpolated_metrics.append(interpolator(target_x))
    
#     return np.array(interpolated_metrics)

# def main():
#     # NORA batch size - this is the key parameter 
#     nora_batch_size = 256
    
#     # Scaling factor to adjust NORA curve - change this to align NORA data
#     # Values > 1.0 stretch the curve more (shifting right)
#     # Values < 1.0 compress the curve (shifting left)
#     # Example: If you want NORA step 100 to align with step 25600*2, use scaling_factor=2.0
#     nora_scaling_factor = 0.3 
#     # Process JSON files (NORA data) with batch size scaling and additional scaling factor
#     json_data, json_steps, json_mean_loss, json_std_loss = process_multiple_files(
#         json_files, nora_batch_size=nora_batch_size, nora_scaling_factor=nora_scaling_factor, 
#         target_points=500, smooth_factor=0.6)
    
#     # Initialize wandb API and fetch metrics
#     api = wandb.Api()
    
#     # Get metrics for wandb projects
#     aligned_metrics1, run_names1, max_length1 = get_project_metrics(wandb_entity1, wandb_project1, record_runs1, metric_name1, 1, invert=False)
#     aligned_metrics2, run_names2, max_length2 = get_project_metrics(wandb_entity2, wandb_project2, record_runs2, metric_name2, 1, invert=False)
#     aligned_metrics3, run_names3, max_length3 = get_project_metrics(wandb_entity2, wandb_project3, record_runs3, metric_name3, 1, invert=False)
#     aligned_metrics4, run_names4, max_length4 = get_project_metrics(wandb_entity2, wandb_project4, record_runs4, metric_name4, 1, invert=False)
    
#     # Target number of points for interpolation
#     target_points = 500
    
#     # Interpolate all metrics to match target points
#     if len(aligned_metrics1) > 0 and max_length1 != target_points:
#         aligned_metrics1 = interpolate_metrics_to_target(aligned_metrics1, max_length1, target_points)
    
#     if len(aligned_metrics2) > 0 and max_length2 != target_points:
#         aligned_metrics2 = interpolate_metrics_to_target(aligned_metrics2, max_length2, target_points)
    
#     if len(aligned_metrics3) > 0 and max_length3 != target_points:
#         aligned_metrics3 = interpolate_metrics_to_target(aligned_metrics3, max_length3, target_points)
    
#     if len(aligned_metrics4) > 0 and max_length4 != target_points:
#         aligned_metrics4 = interpolate_metrics_to_target(aligned_metrics4, max_length4, target_points)
    
#     # Compute mean and variance for wandb projects if data exists
#     mean_metrics1 = np.mean(aligned_metrics1, axis=0) if len(aligned_metrics1) > 0 else np.array([])
#     std_metrics1 = np.std(aligned_metrics1, axis=0) if len(aligned_metrics1) > 0 else np.array([])
    
#     mean_metrics2 = np.mean(aligned_metrics2, axis=0) if len(aligned_metrics2) > 0 else np.array([])
#     std_metrics2 = np.std(aligned_metrics2, axis=0) if len(aligned_metrics2) > 0 else np.array([])
    
#     mean_metrics3 = np.mean(aligned_metrics3, axis=0) if len(aligned_metrics3) > 0 else np.array([])
#     std_metrics3 = np.std(aligned_metrics3, axis=0) if len(aligned_metrics3) > 0 else np.array([])
    
#     mean_metrics4 = np.mean(aligned_metrics4, axis=0) if len(aligned_metrics4) > 0 else np.array([])
#     std_metrics4 = np.std(aligned_metrics4, axis=0) if len(aligned_metrics4) > 0 else np.array([])
    
#     # Create x-axis points that represent epochs for wandb data - restricted to total_epochs
#     x_points = np.linspace(0, total_epochs, target_points)
    
#     # === Create a combined plot ===
#     plt.figure(figsize=(14, 8))
    
#     # Plot wandb data but restrict to total_epochs
#     # Plot first project (blue)
#     if len(mean_metrics1) > 0:
#         plt.plot(x_points, mean_metrics1, label=f"OpenVLA", color="blue", linewidth=1.5)
#         upper_bound1 = mean_metrics1 + std_metrics1
#         lower_bound1 = mean_metrics1 - std_metrics1
#         plt.fill_between(x_points, lower_bound1, upper_bound1, color="blue", alpha=0.2)
    
#     # Plot second project (red)
#     if len(mean_metrics2) > 0:
#         plt.plot(x_points, mean_metrics2, label=f"Diffusion Policy", color="red", linewidth=1.5)
#         upper_bound2 = mean_metrics2 + std_metrics2
#         lower_bound2 = mean_metrics2 - std_metrics2
#         plt.fill_between(x_points, lower_bound2, upper_bound2, color="red", alpha=0.2)
    
#     # Plot third project (green)
#     if len(mean_metrics3) > 0:
#         plt.plot(x_points, mean_metrics3, label=f"ACT", color="green", linewidth=1.5)
#         upper_bound3 = mean_metrics3 + std_metrics3
#         lower_bound3 = mean_metrics3 - std_metrics3
#         plt.fill_between(x_points, lower_bound3, upper_bound3, color="green", alpha=0.2)
    
#     # Plot fourth project (gray)
#     if len(mean_metrics4) > 0:
#         plt.plot(x_points, mean_metrics4, label=r"$\pi_0$", color="gray", linewidth=1.5)
#         upper_bound4 = mean_metrics4 + std_metrics4
#         lower_bound4 = mean_metrics4 - std_metrics4
#         plt.fill_between(x_points, lower_bound4, upper_bound4, color="gray", alpha=0.2)
    
#     # Plot NORA data (JSON files) with its mean and variance
#     if len(json_mean_loss) > 0:
#         # Make sure we're only plotting up to the total_epochs limit
#         valid_indices = np.where(json_steps <= total_epochs)[0]
#         if len(valid_indices) > 0:
#             valid_steps = json_steps[valid_indices]
#             valid_mean_loss = json_mean_loss[valid_indices]
#             valid_std_loss = json_std_loss[valid_indices]
            
#             plt.plot(valid_steps, valid_mean_loss, label="NORA", color="purple", linewidth=1.5)
#             upper_bound_json = valid_mean_loss + valid_std_loss
#             lower_bound_json = valid_mean_loss - valid_std_loss
#             plt.fill_between(valid_steps, lower_bound_json, upper_bound_json, color="purple", alpha=0.2)
    
#     plt.title("Combined Training Loss Comparison")
#     plt.xlabel(f"Epochs (Steps × Batch Size) - Scaling Factor: {nora_scaling_factor}")
#     plt.ylabel("Train Loss")
    
#     # Restrict x-axis to total_epochs
#     plt.xlim(0, total_epochs)
    
#     # Configure legend
#     plt.legend(
#         loc='upper right',
#         prop={'size': 12},
#         frameon=True,
#         shadow=True,
#         fancybox=True
#     )
    
#     # Add grid and adjust layout
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
    
#     # Save the combined plot
#     plt.savefig(f'combined_training_loss_scale_{nora_scaling_factor}.png', dpi=300, bbox_inches='tight')
#     plt.show()

#     # Create a separate plot showing individual NORA (JSON) runs and their mean
#     plt.figure(figsize=(14, 8))
    
#     # Plot individual JSON files with different line styles
#     colors = ["blue", "red", "green"]
#     line_styles = ["--", "-.", ":"]
    
#     for i, (file_name, data) in enumerate(json_data.items()):
#         if 'common_steps' in data and 'common_loss' in data:
#             # Only plot points within the total_epochs limit
#             valid_indices = np.where(data['common_steps'] <= total_epochs)[0]
#             if len(valid_indices) > 0:
#                 valid_steps = data['common_steps'][valid_indices]
#                 valid_loss = data['common_loss'][valid_indices]
                
#                 plt.plot(valid_steps, valid_loss, 
#                          label=f"Run: {file_name}", 
#                          color=colors[i % len(colors)],
#                          linestyle=line_styles[i % len(line_styles)],
#                          alpha=0.7,
#                          linewidth=1.5)
    
#     # Plot the mean of JSON data
#     if len(json_mean_loss) > 0:
#         # Only plot points within the total_epochs limit
#         valid_indices = np.where(json_steps <= total_epochs)[0]
#         if len(valid_indices) > 0:
#             valid_steps = json_steps[valid_indices]
#             valid_mean_loss = json_mean_loss[valid_indices]
#             valid_std_loss = json_std_loss[valid_indices]
            
#             plt.plot(valid_steps, valid_mean_loss, label="Mean", color="black", linewidth=2.5)
#             upper_bound_json = valid_mean_loss + valid_std_loss
#             lower_bound_json = valid_mean_loss - valid_std_loss
#             plt.fill_between(valid_steps, lower_bound_json, upper_bound_json, color="gray", alpha=0.3, label="±1σ Range")
    
#     plt.title(f"NORA: Individual Runs and Mean Loss (Scaling Factor: {nora_scaling_factor})")
#     plt.xlabel("Epochs (Steps × Batch Size)")
#     plt.ylabel("Loss")
#     plt.xlim(0, total_epochs)  # Restrict x-axis to total_epochs
#     plt.legend(loc='upper right', prop={'size': 12})
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(f'nora_runs_with_mean_scale_{nora_scaling_factor}.png', dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == "__main__":
#     main()