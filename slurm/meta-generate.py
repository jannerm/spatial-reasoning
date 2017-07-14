import subprocess

save_path = '../data/mturk_only_global_10dim_8max_2ref/'
vis_path = '../data/mturk_only_global_10dim_8max_2ref_vis/'
start = 200
end = 300
step = 1
dim = 10
mode = 'global'
# only_global = False

for lower in range(start, end, step):
    command = [ 'sbatch', '--qos=tenenbaum', '-c', '2', '--time=1-12:0', '-J', str(lower), 'generate_worlds.py', \
                '--lower', str(lower), '--num_worlds', str(step), '--dim', str(dim), \
                '--save_path', save_path, '--vis_path', vis_path, '--mode', mode]
    # print command
    subprocess.Popen( command )