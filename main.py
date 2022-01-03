from utils.train import train_evaluate_and_save_log

net_name = 'CircleUNet'
plus_opts = [0, 1, 3, 7]
save_name = f'{net_name}_{"".join(list(map(str,plus_opts)))}'
train_evaluate_and_save_log(net_name=net_name, plus_opts=plus_opts, save_name=save_name, note=plus_opts,
                             show_examples=False)