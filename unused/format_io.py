import numpy as np

def format_io(f, args, output_defaults = None):
    df_is_list = type(args['df']) == list
    df = np.array(args['df'], dtype=object)

    if len(df) == 0:
        if output_defaults is not None:
            return df, *output_defaults
        else:
            return df

    out_df, *other_output = f(**args)

    if df_is_list:
        return out_df.tolist() if out_df is not None else None, *other_output

    return out_df, *other_output