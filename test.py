import re
ext4fs_stats = re.compile(r'Created filesystem with ([0-9]+)/([0-9]+)blocks')
line = 'Created filesystem with 123/456 blocks'
m = ext4fs_stats.match(line)
print(m.group(0))