import json
import sympy

def find_psi(w, q, n):
    """
    Find a number psi such that:
    psi^2 = w mod q
    psi^n = -1 mod q
    """
    # Check if q is prime
    if not sympy.isprime(q):
        return None

    # Check if w is a quadratic residue mod q
    if sympy.legendre_symbol(w, q) == 1:
        # Find the square root of w mod q
        psi = sympy.sqrt_mod(w, q, all_roots=False)
        # Verify the second condition
        if pow(psi, n, q) == q - 1:
            return psi
    return None

# File paths
inputFilePath = '/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/prothPrime/test/test.json'  # Replace with your actual input file path
outputFilePath = '/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/prothPrime/test/test.json'  # Replace with your actual output file path
n = 2**3  # Example value, can be set as needed

# Read the JSON data from file
with open(inputFilePath, 'r') as file:
    data = json.load(file)

# Process each entry in the JSON data and add a 'psi' field
filtered_data = []
for entry in data:
    w = entry["w"]
    q = entry["Dec"]
    psi = find_psi(w, q, n)
    if psi is not None:
        entry["psi"] = psi
        filtered_data.append(entry)

# Write the modified data to a JSON file
with open(outputFilePath, 'w') as file:
    json.dump(filtered_data, file, indent=4)
