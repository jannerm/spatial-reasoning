import os, pickle, argparse, subprocess, numpy as np, scipy.misc

def place_goal(img, goal, save_path, num_cells = 10, width = 4):
    img = img.copy()
    dim = img.shape[0]
    cell_size = float(dim) / num_cells
    half_width = width / 2

    ## add black rectangular grid
    for i in range(0, num_cells+1):
        low = int(i * cell_size) - half_width
        high = low + half_width
        img[low:high, :, :] = 0
        img[:, low:high, :] = 0

    if goal != None:
        ## add red square around goal cell
        goal_row_low = int(goal[0] * cell_size)
        goal_row_high = min( int(goal_row_low + cell_size), img.shape[0] - 1)
        goal_col_low = int(goal[1] * cell_size)
        goal_col_high = min( int(goal_col_low + cell_size), img.shape[1] - 1)

        img[max(goal_row_low-width, 0):goal_row_low+width, goal_col_low:goal_col_high+1, :] = [255,0,0]
        img[goal_row_high-width:goal_row_high+width, goal_col_low:goal_col_high+1, :] = [255,0,0]
        img[goal_row_low:goal_row_high+1, max(goal_col_low-width, 0):goal_col_low+width, :] = [255,0,0]
        img[goal_row_low:goal_row_high+1, goal_col_high-width:goal_col_high+width, :] = [255,0,0]

    scipy.misc.imsave(save_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default='../data/mturk_nonunique_global_10dim_8max_2ref/')
    # parser.add_argument('--sprite_path', type=str, default='../data/mturk_nonunique_global_10dim_8max_2ref_vis/')
    parser.add_argument('--data_path', type=str, default='../data/mturk_only_global_10dim_8max_2ref/')
    parser.add_argument('--sprite_path', type=str, default='../data/mturk_only_global_10dim_8max_2ref_vis/')
    parser.add_argument('--save_path', type=str, default='global_trial/')
    parser.add_argument('--start', type=int, default=200)
    parser.add_argument('--end', type=int, default=300)
    parser.add_argument('--dim', type=int, default=10)
    args = parser.parse_args()

    print args, '\n'

    if not os.path.exists(args.save_path):
        subprocess.Popen(['mkdir', args.save_path])
    
    for i in range(args.start, args.end):
        data_path = os.path.join(args.data_path, str(i) + '.p')
        data = pickle.load( open(data_path, 'rb') )

        sprite_path = os.path.join(args.sprite_path, str(i) + '_sprites.png')
        sprites = scipy.misc.imread(sprite_path)

        dump_path = os.path.join(args.save_path, str(i) + '.p')
        pickle.dump(data, open(dump_path, 'wb'))

        save_path = os.path.join(args.save_path, str(i) + '_clean.png')
        place_goal(sprites, None, save_path, num_cells = args.dim)

        for g, goal in enumerate(data['goals']):
            print data['instructions'][g], goal
            save_path = os.path.join(args.save_path, str(i) + '_' + str(g) + '.png')
            print save_path, '\n'
            place_goal(sprites, goal, save_path, num_cells = args.dim)
        









