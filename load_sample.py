import numpy as np

# Load the numpy file (allow_pickle=True for object arrays)
file_path = "/home/james/tda-particle-correlations/data/ensemble_data/stddev_0.2/sample_000_persistence.npy"
data = np.load(file_path, allow_pickle=True)

# Print basic info about the array
print(f"Array shape: {data.shape}")
print(f"Array dtype: {data.dtype}")

# Extract the actual data
if data.shape == ():
    # 0-dimensional array, extract the item
    actual_data = data.item()
    print(f"\nData type: {type(actual_data)}")
    
    if isinstance(actual_data, dict):
        print("\nDictionary keys:", list(actual_data.keys()))
        for key in actual_data.keys():
            value = actual_data[key]
            print(f"\n{key} shape: {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"First 10 rows of {key}:")
                print(np.array(value)[:10])
    elif hasattr(actual_data, '__len__'):
        print("\nFirst 10 rows:")
        print(np.array(actual_data)[:10])
    else:
        print("\nData:")
        print(actual_data)
else:
    print("\nFirst 10 rows:")
    print(data[:10])

# Print max of h0 third column (number of points)
if data.shape == () and isinstance(data.item(), dict):
    actual_data = data.item()
    if 'h0' in actual_data:
        h0_array = np.array(actual_data['h0'])
        max_h0_third_col = np.max(h0_array[:, 2])
        print(f"\nMax of h0 third column: {max_h0_third_col}")

print("\n" + "="*60)
print("EXPLANATION OF THE DATA STRUCTURE:")
print("="*60)
print("\nWhat's happening:")
print("- np.save() was called with a dictionary as the object")
print("- This creates a 0-dimensional numpy array containing the dict")
print("- The dict itself contains the actual numpy arrays (h0, h1, etc.)")
print("\nWhy this might be done:")
print("✓ Convenience: Save multiple related arrays in one .npy file")
print("✓ Simple: No need for additional libraries (like h5py, pickle)")
print("✓ Metadata: Can store scalars (n_points, etc.) alongside arrays")
print("\nDownsides:")
print("✗ Requires allow_pickle=True (security risk with untrusted data)")
print("✗ Not as efficient as native numpy arrays")
print("✗ Loses numpy's memory-mapping capabilities")
print("\nBetter alternatives:")
print("• np.savez() - Save multiple arrays in one file (no pickle needed)")
print("• HDF5 (h5py) - Hierarchical data, better for large datasets")
print("• Direct .npy for each array - Simplest, most efficient")