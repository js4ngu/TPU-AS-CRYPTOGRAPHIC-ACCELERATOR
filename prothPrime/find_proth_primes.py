from sympy import isprime
import math
import json
import torch
import itertools

def find_modulus(veclen: int, minimum: int) -> int:
    if veclen < 1 or minimum < 1:
        raise ValueError()
    start = (minimum - 1 + veclen - 1) // veclen
    for i in itertools.count(max(start, 1)):
        n = i * veclen + 1
        assert n >= minimum
        if is_prime(n):
            return n
    raise AssertionError("Unreachable")
'''
def is_prime(n: int) -> bool:
    if n <= 1:
        raise ValueError()
    return all((n % i != 0) for i in range(2, sqrt(n) + 1))
'''
def sqrt(n: int) -> int:
    if n < 0:
        raise ValueError()
    i = 1
    while i * i <= n:
        i *= 2
    result = 0
    while i > 0:
        if (result + i)**2 <= n:
            result += i
        i //= 2
    return result

def find_primitive_root(degree: int, totient: int, mod: int) -> int:
    if not (1 <= degree <= totient < mod):
        raise ValueError()
    if totient % degree != 0:
        raise ValueError()
    gen = find_generator(totient, mod)
    root = pow(gen, totient // degree, mod)
    assert 0 <= root < mod
    return root

def find_generator(totient: int, mod: int) -> int:
    if not (1 <= totient < mod):
        raise ValueError()
    for i in range(1, mod):
        if is_primitive_root(i, totient, mod):
            return i
    raise ValueError("No generator exists")

def is_primitive_root(val: int, degree: int, mod: int) -> bool:
    if not (0 <= val < mod):
        raise ValueError()
    if not (1 <= degree < mod):
        raise ValueError()
    pf = unique_prime_factors(degree)
    # pf를 리스트로 변환하고 반복
    return pow(val, degree, mod) == 1 and all((pow(val, degree // p, mod) != 1) for p in pf.tolist())


def unique_prime_factors(n: int) -> torch.Tensor:
    if n < 1:
        raise ValueError()
    result = []
    i = 2
    end = sqrt(n)
    while i <= end:
        if n % i == 0:
            n //= i
            result.append(i)
            while n % i == 0:
                n //= i
            end = sqrt(n)
        i += 1
    if n > 1:
        result.append(n)
    return torch.tensor(result)

# Given values
smallBit = 1
largeBit = 10
n = 8
file_path = "/home/jongsang/File/tpu-as-crypto/TPU-AS-CRYPTOGRAPHIC-ACCELERATOR/prothPrime/test/test2.json"

# Initialize an empty list to store the prime numbers
prime_numbers = []
w = []
# Determine the range of k based on the given range for the numbers
k_min = 1  # k should be a positive integer
k_max = (2**largeBit - 1) // (2**n)  # derived from the inequality k * 2^n + 1 <= 2^largeBit

# Iterate through the possible values of k and check if the resulting number is prime
for k in range(k_min, k_max + 1):
    print(k/(k_max - k_min + 1) * 100, "%")
    number = k * 2**n + 1
    if isprime(number) and 2**smallBit <= number <= 2**largeBit:
        prime_numbers.append((k, number))
        w.append(find_primitive_root(2**n, number - 1, number))
print(w)
print(prime_numbers)
# 소수 개수 계산
total_primes = len(prime_numbers)

# output 리스트를 생성하는 동안 진행률을 로깅
output = []
for idx, (k, number) in enumerate(prime_numbers):
    progress = (idx + 1) / total_primes * 100  # 현재 진행률 계산
    print(f"Processing {idx + 1}/{total_primes} ({progress:.2f}%)")  # 진행률 출력
    
    # output 리스트에 항목 추가
    output.append({"BitLength": int(math.log2(number)) + 1,
                   "w": w[idx],
                   "Dec":number,
                   "chk":w[idx]**(2**n) % number,
                   "MathExpression": f"{k} * 2^{n} + 1",
                   "HexRepresentation": hex(number)})

# JSON 파일에 output 리스트 저장
with open(file_path, "w") as json_file:
    json.dump(output, json_file, indent=4)

print(f"JSON data has been saved to {file_path}")
