import os

root = r'E:\GrainQuality\tiny_data'

print("Contents of tiny_data:")
for item in os.listdir(root):
    full = os.path.join(root, item)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"  📁 {item}/ — {len(files)} items")
        # Show first 3 files inside
        for f in files[:3]:
            print(f"       {f}")
    else:
        print(f"  📄 {item}")