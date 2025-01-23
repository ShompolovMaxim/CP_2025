def is_k_anonimus_v0(df, k):
    for current_row in df:
        kol = 0
        for row in df:
            kol += (row == current_row).all()
        if kol < k:
            return False
    return True

def is_k_anonimus(df, k):
    counts = dict()
    for row in df:
        if tuple(row.tolist()) in counts:
            counts[tuple(row.tolist())] += 1
        else:
            counts[tuple(row.tolist())] = 1
    for _, count in counts.items():
        if count < k:
            return False
    return True