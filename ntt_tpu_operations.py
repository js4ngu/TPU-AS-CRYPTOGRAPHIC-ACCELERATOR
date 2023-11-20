import torch
import itertools
import numpy as np
import random
import gc
import time
import json

# Parameter Read
def readJson(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def LoadParameter(file_path, matrix_type):
    data = readJson(file_path)
    
    if matrix_type in ['NTT', 'INTT', 'N', 'w', 'invW', 'invN', 'p']:
        return data.get(matrix_type)
    else:
        raise ValueError("Invalid matrix_type. Choose among 'NTT', 'INTT', 'N', 'w', 'invW', or 'invN'.")

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

def is_prime(n: int) -> bool:
    if n <= 1:
        raise ValueError()
    return all((n % i != 0) for i in range(2, sqrt(n) + 1))

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

def find_params_Elment(veclen, minmod: int) -> torch.Tensor:
    mod = find_modulus(veclen, minmod)
    root = find_primitive_root(veclen, mod - 1, mod)
    return torch.tensor((root, mod))

def find_params(vec0: torch.Tensor, vec1: torch.Tensor) -> torch.Tensor:
    # Calculate the max value and minimum modulus
    maxval = torch.max(vec0.max(), vec1.max())
    minmod = int(maxval**2 * len(vec0) + 1)
    root, mod = find_params_Elment(vec0, minmod)
    return torch.tensor((root, mod))



def toRNS(tensor, moduli):
    degree = len(tensor)
    RNS = torch.zeros((degree), dtype=torch.int64)

    for j in range(degree):   # 두 번째 반복문은 modSize 범위
        RNS[j] = tensor[j] % moduli  # 인덱스 순서 변경
    return RNS

def operandMatrixV2(aRNS, bRNS):
    out = torch.stack([aRNS, bRNS], dim = 0)
    print(out.dtype)

    return out

def element_wise_multV3(w, N, p):
    W = torch.zeros((N), dtype=torch.int64)  # Set the result tensor type to int32
    print(W)
    for j in range(N):
        # Convert to int for intermediate multiplication to avoid overflow
        value = int(w[0,j]) * int(w[1,j])
        value = value % p
        W[j] = value 
    return W

# 확장된 유클리드 알고리즘을 사용해 m_i의 모듈러 역수를 찾는 함수
def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = extended_gcd(b % a, a)
        return (g, y - (b // a) * x, x)

def modinv(a, m):
    g, x, y = extended_gcd(a, m)
    return x % m

def CRTv3(n, a):
    # Convert tensors to lists of Python native int for large number arithmetic
    a = a.clone().detach().tolist()

    total_sum = 0
    prod = 1
    for n_i in n:
        prod *= n_i

    for n_i, a_i in zip(n, a):
        p = prod // n_i
        total_sum += a_i * modinv(p, n_i) * p
    return total_sum % prod

def convert_crt_matV3(INTT,moduli,degree):
      m = len(moduli)
      #TEMP = torch.zeros((m, degree), dtype=torch.int64)
      #TEMP = INTT[:m,degree-1]
      TEMP = INTT[:m,:degree]
      print("Sliece INTT (TEMP)")
      print(TEMP.shape)
      print(TEMP)
      W = torch.zeros((degree, m), dtype=torch.int64)
      for i in range(degree):
        W[i,:m] = TEMP[:, i]
      return W

def tensorCRTv2(a, moduli, degree):
  W = list()#torch.zeros(degree, dtype=torch.int64)
  for i in range(degree):
    W.append(CRTv3(moduli, a[i]))
  return W

def RandomPoly(length, bit_range):
    min_val = 2 ** (bit_range[0] - 1)
    max_val = 2 ** bit_range[1] - 1
    random_values = [random.randint(min_val, max_val) for _ in range(length)]
    return random_values

def paddedRandomPoly(length, bit_range):
    min_val = 2 ** (bit_range[0] - 1)
    max_val = 2 ** bit_range[1] - 1
    half_length = length // 2
    random_values = [random.randint(min_val, max_val) for _ in range(half_length)]
    zero_padding = [0] * (length - half_length)
    
    return random_values + zero_padding

def polyMult(a,b,degreeSize):
    res = [0] * degreeSize
    for i in range(degreeSize//2):
        for j in range(degreeSize//2):
            res[i + j] += a[i] * b[j]
    return res

def convolution(a, b, q=None, mode='linear'):
    if mode == 'linear':
        result_length = len(a) + len(b) - 1
    else:  # For PWC or NWC, the result length is the same as the polynomials
        result_length = len(a)
    
    result = [0] * result_length
    
    for k in range(result_length):
        if mode == 'linear':
            # Compute linear convolution
            for i in range(max(0, k + 1 - len(b)), min(k + 1, len(a))):
                result[k] += a[i] * b[k - i]
        else:
            # Compute PWC or NWC which wraps around the convolution
            for i in range(max(0, k + 1 - len(b)), len(a)):
                wrap_around_index = (k - i) % len(b)
                contribution = a[i] * b[wrap_around_index]
                if mode == 'negative' and i > k:
                    contribution = -contribution
                result[k] += contribution

            # Apply modular arithmetic if q is provided
            if q is not None:
                result[k] %= q

    return result

def dynamicrange(moduli):
    result = 1
    for num in moduli:
        result *= num
    return result

def cmp(res, baseline, degreeSize):
    if res == baseline:
        print("PASS")
    else : 
        print("FAIL")
    '''
        for i in range(degreeSize):
            if baseline[i] != res[i]:
                print("[",i,"] Elment is != (baseline/res) : ",baseline[i],"/",res[i])
    '''
def peSel(size):
    if size == 8:
        return torch.int8
    elif size == 16:
        return torch.int16
    elif size == 32:
        return torch.int32
    elif size == 64:
        return torch.int64

def RNSbasedNTT(a, b, w, p, degreeSize, moduli, filepath, peSize):
    #peSize=torch.int64
    #input polynomial
    a = torch.tensor(a)
    b = torch.tensor(b)
    aRNS = toRNS(a, moduli)
    bRNS = toRNS(b, moduli)
    opMat = operandMatrixV2(aRNS, bRNS)
    print("opMAT SHAPE : ", opMat.shape)
    print(opMat)
    
    #setting NTT matrix``
    nttMat = torch.tensor(LoadParameter(filepath, "NTT"), dtype=peSize)
    print("nttMat SHAPE : ", nttMat.shape)
    print(nttMat)

    #NTT
    NTT = torch.matmul(opMat, nttMat) % p# = strassen(opMat, nttMat, peSize, 0) % p
    print("NTT SHAPE : ", NTT.shape)
    print(NTT)
    print(NTT[1][3])
    #point wise mult
    MULT = element_wise_multV3(NTT, degreeSize, p)
    print("MULT SHAPE : ", MULT.shape)
    print(MULT)
    
    #setting intt mat
    invMat = torch.tensor(LoadParameter(filepath, "INTT"), dtype=peSize)
    print("invMat SHAPE : ", invMat.shape)
    print(invMat)

    #INTT
    INTT = torch.matmul(MULT, invMat) % p#strassen(MULT, invMat, peSize, 0) % p
    print("INTT SHAPE : ", INTT.shape)
    print(INTT)

    return INTT

def rnsNTTbasedPolymult(a, b, targertP, mode, w, moduli, degreeSize, peSize, dir):
    #merged each rns ch to 1 mat
    peSize = peSel(peSize) #torch.int64
    RNS = torch.zeros((len(moduli),degreeSize), dtype=peSize)
    for i in range(0,len(moduli)):
        print("Status :", i/len(w) * 100, "%")
        filepath = dir + str(moduli[i]) + ".json"
        RNS[i] = RNSbasedNTT(a, b, w[i], moduli[i], degreeSize, moduli[i], filepath, peSize)

    #CRT
    res = convert_crt_matV3(RNS, moduli, degreeSize)
    out = tensorCRTv2(res, moduli, degreeSize)
    #print("a : ", a)
    #print("b : ", b)
    #print("crt result", out)

    baseline = convolution(a, b, targertP, mode)
    #print("baseline", baseline)
    cmp(out, baseline, degreeSize)
