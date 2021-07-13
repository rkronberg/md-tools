def get_blocks(u, n_blocks):
    '''Divides trajectory into blocks.

    Args:
        u: MDAnalysis Universe object containing input trajectory.
        n_blocks: Number of blocks trajectory is split into.

    Returns:
        List of ranges specifying the indices forming each block.
    '''

    print('Calculating number of frames: ', end='\r')
    n_frames = u.trajectory.n_frames
    print('Calculating number of frames: %i' % n_frames)

    n_blocks = n_blocks
    frames_per_block = n_frames//n_blocks
    blocks = [range(i*frames_per_block, (i+1)*frames_per_block)
              for i in range(n_blocks-1)]
    blocks.append(range((n_blocks-1)*frames_per_block, n_frames))

    return blocks
