import json
import math
import sys

def calcDlange(dir, min_bit_length, max_bit_length, P):
    with open(dir, 'r') as file:
        proth_primes = json.load(file)

    product = 1
    for prime_info in proth_primes:
        if min_bit_length <= prime_info['BitLength'] <= max_bit_length:
            product *= prime_info['Dec']
    print(min_bit_length,"~",max_bit_length,"bit can cover target P : ", product > P)
    print("Dynamic Range: ", product.bit_length(), "bits")

def main():
    dir = '/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/prothPrime/FHE/15.json'

    min_bit_length = 17
    targetP = 2**435 - 2**33 + 1
    for i in range(min_bit_length,27):
        max_bit_length = i
        calcDlange(dir, min_bit_length, max_bit_length, targetP)

main()  # 이 예제는 수정된 함수를 사용하여 실행합니다.