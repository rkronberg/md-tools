'''
Divides trajectory frames into as many blocks
as the number of requested parallel jobs.
'''


def get_blocks(u, n_jobs):

    print('Calculating number of frames: ', end='\r')
    n_frames = u.trajectory.n_frames
    print('Calculating number of frames: %i' % n_frames)

    n_blocks = n_jobs
    frames_per_block = n_frames//n_blocks
    blocks = [range(i*frames_per_block, (i+1)*frames_per_block)
              for i in range(n_blocks-1)]
    blocks.append(range((n_blocks-1)*frames_per_block, n_frames))

    return blocks
