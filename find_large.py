import os

threshold = 50 * 1024 * 1024  # 50MB
large_files = []

for root, dirs, files in os.walk('.'):
    if '.git' in root:
        continue
    for f in files:
        filepath = os.path.join(root, f)
        try:
            size = os.path.getsize(filepath)
            if size > threshold:
                large_files.append((filepath, size))
        except OSError:
            pass

large_files.sort(key=lambda x: x[1], reverse=True)
with open('large_files.txt', 'w') as f:
    for filepath, size in large_files:
        f.write(f"{filepath}: {size / (1024 * 1024):.2f} MB\n")
