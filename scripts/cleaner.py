import pandas as pd

cols_to_keep = ['name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc',
                'condition_code', 'n_obs_used', 'H', 'neo', 'pha', 'diameter',
                'albedo', 'rot_per', 'moid', 'class', 'n', 'per', 'ma']


def clean(main_csv):
    df = pd.read_csv(main_csv, header=0, usecols=cols_to_keep)
    df.drop(df[df['a'] < 0].index, inplace=True)
    au_array = ['a', 'q', 'ad', 'moid']
    conv_factor = 1.496e8
    for i in au_array:
        df[i] = conv_factor * df[i]
    df.to_csv("Asteroid_au_to_km.csv")


if __name__ == '__main__':
    clean("Asteroid_Updated.csv")


