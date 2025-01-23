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

def is_l_diverse(quasi_identifiers, sensitives, k, l):
    counts = dict()
    sensitive_values = dict()
    for identifiers, sensitive in zip(quasi_identifiers, sensitives):
        if tuple(identifiers.tolist()) in counts:
            counts[tuple(identifiers.tolist())] += 1
            sensitive_values[tuple(identifiers.tolist())].add(sensitive)
        else:
            counts[tuple(identifiers.tolist())] = 1
            sensitive_values[tuple(identifiers.tolist())] = set()
            sensitive_values[tuple(identifiers.tolist())].add(sensitive)
    for key in counts.keys():
        if counts[key] < k:
            return False
        if len(sensitive_values[key]) < l:
            return False
    return True