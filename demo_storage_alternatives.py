"""
Demonstration of different ways to store multiple arrays
Shows current approach vs. better alternatives
"""
import numpy as np
import os

print("="*70)
print("DEMONSTRATION: Storage Methods for Multiple Arrays")
print("="*70)

# Create some sample data similar to persistence data
h0 = np.random.rand(100, 3)
h1 = np.random.rand(50, 3)
n_points = 692
max_edge_length = 2.5

print("\n1. CURRENT APPROACH: Dictionary in .npy (requires pickle)")
print("-" * 70)
data_dict = {
    'h0': h0,
    'h1': h1,
    'n_points': n_points,
    'max_edge_length': max_edge_length
}
np.save('demo_dict.npy', data_dict, allow_pickle=True)
print("Saved: np.save('demo_dict.npy', data_dict, allow_pickle=True)")
print("Loading requires: np.load('demo_dict.npy', allow_pickle=True).item()")
print("✗ Security risk: allow_pickle=True")
print("✗ Creates 0-dimensional array wrapper")
file_size_1 = os.path.getsize('demo_dict.npy')
print(f"File size: {file_size_1:,} bytes")

print("\n2. BETTER APPROACH: np.savez() (no pickle needed)")
print("-" * 70)
np.savez('demo_savez.npz', 
         h0=h0, 
         h1=h1, 
         n_points=n_points,
         max_edge_length=max_edge_length)
print("Saved: np.savez('demo_savez.npz', h0=h0, h1=h1, ...)")
print("Loading: data = np.load('demo_savez.npz')")
print("Access: data['h0'], data['h1'], etc.")
print("✓ No pickle required - safer")
print("✓ Native numpy format")
print("✓ Can load individual arrays without loading all")
file_size_2 = os.path.getsize('demo_savez.npz')
print(f"File size: {file_size_2:,} bytes")

# Demonstrate loading
data = np.load('demo_savez.npz')
print(f"\nLoaded keys: {list(data.keys())}")
print(f"h0 shape: {data['h0'].shape}")
print(f"n_points: {data['n_points']}")

print("\n3. COMPRESSED VERSION: np.savez_compressed()")
print("-" * 70)
np.savez_compressed('demo_compressed.npz',
                    h0=h0,
                    h1=h1,
                    n_points=n_points,
                    max_edge_length=max_edge_length)
print("Saved: np.savez_compressed('demo_compressed.npz', ...)")
print("✓ All benefits of savez()")
print("✓ Compressed - smaller file size")
file_size_3 = os.path.getsize('demo_compressed.npz')
print(f"File size: {file_size_3:,} bytes ({100*file_size_3/file_size_2:.1f}% of uncompressed)")

print("\n4. INDIVIDUAL FILES (simplest, most flexible)")
print("-" * 70)
np.save('demo_h0.npy', h0)
np.save('demo_h1.npy', h1)
print("Saved: np.save('demo_h0.npy', h0)")
print("       np.save('demo_h1.npy', h1)")
print("✓ Simplest approach")
print("✓ Most efficient for large arrays")
print("✓ Can memory-map for huge datasets")
print("✓ Easy to version control which arrays changed")
print("✗ Requires multiple files")
file_size_4 = os.path.getsize('demo_h0.npy') + os.path.getsize('demo_h1.npy')
print(f"Total file size: {file_size_4:,} bytes")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\nFor your use case (persistence data), I recommend:")
print("  → np.savez() or np.savez_compressed()")
print("\nWhy:")
print("  • Multiple related arrays in one file")
print("  • No security concerns (no pickle)")
print("  • Native numpy format")
print("  • Easy to load and use")
print("\nSimple migration:")
print("  Replace: np.save(file, dict, allow_pickle=True)")
print("  With:    np.savez(file, **dict)  # unpacks dict as kwargs")

# Cleanup demo files
for f in ['demo_dict.npy', 'demo_savez.npz', 'demo_compressed.npz', 
          'demo_h0.npy', 'demo_h1.npy']:
    if os.path.exists(f):
        os.remove(f)
print("\n(Demo files cleaned up)")

