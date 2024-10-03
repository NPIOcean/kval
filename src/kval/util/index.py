'''
KVAL.UTIL.INDEX

Various functions related ti slicing/indexing
'''

def indices_to_slices(indices):

    if len(indices) == 0:
        return []

    slices = []
    start = indices[0]

    for i in range(1, len(indices)):
        # Check if the current index is not consecutive
        if indices[i] != indices[i-1] + 1:
            # If the slice has length 1, append the single index
            if start == indices[i-1]:
                slices.append(int(start))
            else:
                slices.append(slice(int(start), int(indices[i-1] + 1)))
            start = indices[i]

    # Handle the last slice
    if start == indices[-1]:
        slices.append(int(start))
    else:
        slices.append(slice(int(start), int(indices[-1] + 1)))

    return slices
