import argparse

p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--shout', action='store_true')
a = p.parse_args()
msg = f'Hello, {a.name}'
print(msg.upper() if a.shout else msg)
