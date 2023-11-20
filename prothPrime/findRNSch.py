import json
from itertools import combinations
import math

def find_closest_combination(dec_values, target):
    closest_combination = None
    closest_value = float('inf')
    total_combinations = sum(math.comb(len(dec_values), i) for i in range(1, len(dec_values)+1))
    combinations_processed = 0

    for length in range(len(dec_values), 0, -1):
        for subset in combinations(dec_values, length):
            combinations_processed += 1
            product_value = math.prod(subset)
            if target <= product_value < closest_value:
                closest_value = product_value
                closest_combination = subset
            if closest_value == target:
                return closest_combination, closest_value

            if combinations_processed % 100000 == 0:
                print(f"Processed {combinations_processed/total_combinations * 100} % ")

    return closest_combination, closest_value

def findRNSch(json_data, target, min_bitlength, max_bitlength):
    # Filter items based on BitLength range
    filtered_data = [item for item in json_data if min_bitlength <= item['BitLength'] <= max_bitlength]
    dec_values = [item['Dec'] for item in filtered_data]
    return find_closest_combination(dec_values, target)

def main():
    file_path = '/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/prothPrime/FHE/15.json'
    output_path = '/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/prothPrime/FHE/out15.json'
    target = 2**435 - 2**33 + 1
    min_bitlength = 16  # Set your minimum bitlength
    max_bitlength = 26  # Set your maximum bitlength

    with open(file_path, 'r') as f:
        json_data = json.load(f)

    closest_combination, closest_value = findRNSch(json_data, target, min_bitlength, max_bitlength)
    print("Closest combination:", closest_combination)
    print("Closest value:", closest_value)
    print("Target Value:", target)
    print("Closest value - Target Value:", closest_value - target)

    output = {
        "Closest combination": list(closest_combination),
        "Closest Value": closest_value,
        "Target P": target,
        "Closest value - Target Value": closest_value - target
    }

    with open(output_path, "w") as json_file:
        json.dump(output, json_file, indent=4)

    print(f"JSON data has been saved to {output_path}")

main()
