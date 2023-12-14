import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='ICEWS05')
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--hidden_dim', type=int, default=200)
args.add_argument('--gpu', type=int, default=1)
args.add_argument('--batch_size', type=int, default=1024)
args.add_argument('--comb_model', type=int, default=1)

args = args.parse_args()
print(args)